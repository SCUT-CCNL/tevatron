import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from .loss import SimpleContrastiveLoss, DistributedContrastiveLoss, SimpleContrastiveLoss4Splade

import logging

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        # inputs[0]: query data->{’input_ids‘: a tensor; 'attention_mask': a tensor};
        # inputs[1]: passage data->{’input_ids‘: a tensor; 'attention_mask': a tensor}
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))  # equal to {k: v.to(args.device) for k, v in x.items()}
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        logging.info(f"num_workers:{self.args.dataloader_num_workers}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        """
        args:
            model (`nn.Module`):
                    The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
        """
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {'query': queries}, {'passage': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor


class ELKTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super(ELKTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

        logger.info("Saving all params in the dense model to %s", output_dir)
        # save entity embeddings
        logger.info("save entity embeddings to %s", os.path.join(output_dir, "trained_ent_embeddings_100d.pt"))
        torch.save(self.model.embed, os.path.join(output_dir, "trained_ent_embeddings_100d.pt"))
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        state_dict = model_to_save.state_dict()
        # save other params except entity embeddings
        state_dict.pop("embed.weight")
        torch.save(state_dict, os.path.join(output_dir, "pytorch_dense_model.bin"))

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepare = super()._prepare_inputs(inputs)
        for x in prepare:
            x['input_ent'] = self.model.embed(x['input_ent'] + 1)

        return prepare

