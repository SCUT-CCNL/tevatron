import copy
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel, AutoConfig
from .encoder import EncoderPooler, EncoderOutput

from tevatron.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments
from knowledge_bert.modeling import BertModel, BertConfig, gelu, BertLayerNorm

import logging

logger = logging.getLogger(__name__)


class ELKPooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True):
        super(ELKPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None, **kwargs):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError


def save_model(path: str, model: nn.Module):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(path, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def save_config(path: str, conf: BertConfig):
    if conf is not None:
        with open(os.path.join(path, 'elk_config.json'), 'w') as f:
            json.dump(conf.to_dict(), f)


def reformat_state_dict(path: str):
    state_dict = torch.load(path)
    st = {}
    for k in state_dict:
        name = "bert." + k
        st[name] = state_dict[k]
    del state_dict
    return st


def dot_attention(q, k, v, bias=None):
    attn_weights = torch.matmul(q, k.transpose(2, 1))
    if bias is not None:
        attn_weights = attn_weights + bias
    attn_weights = F.softmax(attn_weights, -1)
    output = torch.matmul(attn_weights, v)
    return output


class ELKModel(nn.Module):
    TRANSFORMER_CLS_ELK = BertModel
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 idf: nn.Embedding = None,
                 embed: nn.Embedding = None,
                 pooler: nn.Module = None,
                 qry_use_elk: bool = False,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 temperature: float = 1.0,
                 elk_config: BertConfig = None,
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.idf = idf
        self.embed = embed
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction="mean")
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        self.qry_use_elk = qry_use_elk
        self.temperature = temperature
        self.elk_config = elk_config
        self.ent_dense768 = nn.Linear(100, 768)
        self.ent_act_fn = gelu
        self.ent_LayerNorm = BertLayerNorm(768, eps=1e-12)
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        """
        query: {
            "input_ids": tensor[bs, qry_max_len],
            "attention_mask": tensor[bs, qry_max_len],
            "input_ent": tensor[bs, qry_max_len， ent_dim],
            "ent_mask": tensor[bs, qry_max_len],
        }
        passage: {
            "input_ids": tensor[bs * train_n_passage, psg_max_len],
            "attention_mask": tensor[bs * train_n_passage, psg_max_len],
            "input_ent": tensor[bs * train_n_passage, psg_max_len, ent_dim],
            "ent_mask": tensor[bs * train_n_passage, psg_max_len],
        }
        """
        q_reps, q_hidden, q_ent_hidden = self.encode_query(query)
        p_reps, p_hidden, p_ent_hidden = self.encode_passage(passage)

        if q_reps is not None:
            q_ent_hidden = self.ent_act_fn(q_ent_hidden)
            q_ent_hidden = self.ent_LayerNorm(q_ent_hidden)
        if p_reps is not None:
            p_ent_hidden = self.ent_act_fn(p_ent_hidden)
            p_ent_hidden = self.ent_LayerNorm(p_ent_hidden)

        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            loss_wsim = self.compute_entity_similarity_loss(query["ent_mask"], q_hidden, q_ent_hidden,
                                                            passage["ent_mask"], p_hidden, p_ent_hidden)

            scores = self.compute_similarity(q_reps, p_reps)
            """ 增加softmax温度因子 """
            if self.temperature < 1:
                scores = scores / self.temperature
            scores = scores.view(q_reps.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)

            loss = loss + 0.3 * loss_wsim
            if self.negatives_x_device:
                loss = loss * self.world_size
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, psg):
        if psg is None:
            return None, None, None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out["last_hidden_state"]
        p_ent_hidden = psg_out["last_ent_hidden_state"]
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)
        else:
            p_reps = p_hidden[:, 0]

        p_ent_mask = psg["ent_mask"].unsqueeze(dim=2)
        p_ent_hidden = self.ent_dense768(p_ent_hidden)

        p_ent_hidden = p_ent_mask * p_ent_hidden
        p_ent_bias = (psg["ent_mask"] - 1) * 9999999.0
        p_ent_reps = dot_attention(p_reps.unsqueeze(dim=1), p_ent_hidden, p_ent_hidden, p_ent_bias.unsqueeze(dim=1))
        p_ent_reps = p_ent_reps.squeeze(dim=1)

        p_reps_with_ent = torch.concat((p_reps, p_ent_reps), dim=1)
        assert p_reps_with_ent.dtype == torch.float32
        return p_reps_with_ent, p_hidden, p_ent_hidden

    def encode_query(self, qry):
        if qry is None:
            return None, None, None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out["last_hidden_state"]
        q_ent_hidden = qry_out["last_ent_hidden_state"]
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]

        q_ent_mask = qry["ent_mask"].unsqueeze(dim=2)
        q_ent_hidden = self.ent_dense768(q_ent_hidden)
        q_ent_reps = (q_ent_mask * q_ent_hidden).sum(dim=1) / (q_ent_mask.count_nonzero(dim=1) + 0.00001)

        q_ent_reps_scaled = 1.0 * q_ent_reps
        q_reps_with_ent = torch.concat((q_reps, q_ent_reps_scaled), dim=1)
        assert q_reps_with_ent.dtype == torch.float32
        return q_reps_with_ent, q_hidden, q_ent_hidden

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_entity_similarity_loss(self, q_mask, q_hidden, q_ent_hidden, p_mask, p_hidden, p_ent_hidden):
        p_ent_mask = p_mask.unsqueeze(dim=2)
        p_hidden = p_hidden * p_ent_mask
        p_ent_hidden = p_ent_hidden * p_ent_mask

        q_ent_mask = q_mask.unsqueeze(dim=2)
        q_hidden = q_hidden * q_ent_mask
        q_ent_hidden = q_ent_hidden * q_ent_mask
        loss_wsim = self.mse(p_hidden, p_ent_hidden) + self.mse(q_hidden, q_ent_hidden)
        return loss_wsim

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @staticmethod
    def build_pooler(model_args):
        pooler = ELKPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = ELKPooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            embed: nn.Embedding,
            idf: nn.Embedding,
            **hf_kwargs,
    ):
        elk_config = None
        if model_args.untie_encoder:
            _qry_model_path = model_args.model_name_or_path
            _psg_model_path = model_args.elk_model_path
            logger.info(f'loading query model weight from {_qry_model_path}')
            if not model_args.qry_use_elk:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
            else:
                lm_q, _ = cls.TRANSFORMER_CLS_ELK.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
            logger.info(f'loading passage model weight from {_psg_model_path}')
            lm_p, _ = cls.TRANSFORMER_CLS_ELK.from_pretrained(
                _psg_model_path,
                **hf_kwargs
            )
            elk_config = BertConfig.from_json_file(os.path.join(_psg_model_path, "elk_config.json"))
        else:
            logger.info(f"share parameters between qry encoder & psg encoder")
            logger.info(f'loading qry/psg model weight from {model_args.elk_model_path}')
            lm_q, _ = cls.TRANSFORMER_CLS_ELK.from_pretrained(model_args.elk_model_path, **hf_kwargs)
            lm_p = lm_q
            elk_config = BertConfig.from_json_file(os.path.join(model_args.elk_model_path, "elk_config.json"))

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            idf=idf,
            embed=embed,
            pooler=pooler,
            qry_use_elk=model_args.qry_use_elk,
            untie_encoder=model_args.untie_encoder,
            negatives_x_device=train_args.negatives_x_device,
            temperature=train_args.temperature,
            elk_config=elk_config,
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            untie_encoder,
            qry_use_elk,
            **hf_kwargs,
    ):
        lm_q = None
        lm_p = None
        if untie_encoder:
            if os.path.isdir(model_name_or_path):
                _qry_model_path = os.path.join(model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
                if os.path.exists(_qry_model_path):
                    logger.info(f'found separate weight for query/passage encoders')
                    logger.info(f'loading query model weight from {_qry_model_path}')
                    if not qry_use_elk:
                        lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                            _qry_model_path,
                            **hf_kwargs
                        )
                    else:
                        st = reformat_state_dict(os.path.join(_qry_model_path, "pytorch_model.bin"))
                        lm_q, _ = cls.TRANSFORMER_CLS_ELK.from_pretrained(
                            _qry_model_path,
                            state_dict=st,
                            **hf_kwargs
                        )
                    logger.info(f'loading passage model weight from {_psg_model_path}')
                    st = reformat_state_dict(os.path.join(_psg_model_path, "pytorch_model.bin"))
                    lm_p, _ = cls.TRANSFORMER_CLS_ELK.from_pretrained(
                        _psg_model_path,
                        state_dict=st,
                        **hf_kwargs
                    )
                else:
                    logger.info(f"{_qry_model_path} not exists, please use local paths to load qry_encoder && "
                                f"psg_encoder")
                    exit("1")
            else:
                logger.info(f"{model_name_or_path} not exists, please use local paths to load qry_encoder "
                            f"&& psg_encoder")
                exit("1")
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            st = reformat_state_dict(os.path.join(model_name_or_path, "pytorch_model.bin"))
            lm_q, _ = cls.TRANSFORMER_CLS_ELK.from_pretrained(model_name_or_path, state_dict=st, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)

            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            idf=None,
            embed=None,
            pooler=pooler,
            qry_use_elk=qry_use_elk,
            untie_encoder=untie_encoder,
        )

        state_dict = torch.load(os.path.join(model_name_or_path, "pytorch_dense_model.bin"))
        for param_key in list(state_dict):
            if not param_key.startswith("ent"):
                state_dict.pop(param_key)
        model.load_state_dict(state_dict, strict=False)
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            query_path = os.path.join(output_dir, 'query_model')
            passage_path = os.path.join(output_dir, 'passage_model')
            os.makedirs(query_path)
            os.makedirs(passage_path)

            if not self.qry_use_elk:
                self.lm_q.save_pretrained(query_path)
            else:
                save_model(query_path, self.lm_q)
                save_config(query_path, self.elk_config)

            save_model(passage_path, self.lm_p)
            save_config(passage_path, self.elk_config)
        else:
            save_model(output_dir, self.lm_p)
            save_config(output_dir, self.elk_config)
        if self.pooler:
            self.pooler.save_pooler(output_dir)
