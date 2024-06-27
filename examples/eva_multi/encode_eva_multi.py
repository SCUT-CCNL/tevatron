import logging
import os
import pickle
import sys
import time
from contextlib import nullcontext

import numpy as np
from knowledge_bert import BertTokenizer
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertConfig
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeDatasetForERNIE, EncodeCollatorForERNIE
from tevatron.modeling import EncoderOutput, EVAMultiModel
from tevatron.datasets import HFQueryDataset, HFCorpusDataset

logger = logging.getLogger(__name__)

def prune_passage_representation(psg_reps, rep_nums_list, pid_lookup):
    """
    psg_reps: [psg_num, max_psg_cluster_num, dim]
    rep_nums_list: like [2, 4, 1], rep_nums_list表示psg_reps[0]中有效的文档表征数量
    """
    max_psg_cluster_num = psg_reps.shape[1]
    psg_reps = psg_reps.reshape(psg_reps.shape[0] * psg_reps.shape[1], psg_reps.shape[2])
    idx = []
    new_lookup = []
    doc_num = 0
    for num, pid in zip(rep_nums_list, pid_lookup):
        # 1. 根据每个文档的聚簇数量保留相应个数的文档表征
        if num == 0:
            num = 1
        base = doc_num * max_psg_cluster_num
        tmp = [i + base for i in range(num)]
        idx += tmp
        doc_num += 1
        # 2. 构造新的lookup表
        new_lookup.extend([pid] * num)
    # print(f"idx: {idx}")
    idx = torch.tensor(idx)
    return psg_reps[idx], new_lookup


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    model_args.num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = EVAMultiModel.load(
        model_name_or_path=model_args.model_name_or_path,
        untie_encoder=model_args.untie_encoder,
        cache_dir=model_args.cache_dir,
    )

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=tokenizer, data_args=data_args,
                                        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    start_time = time.time()
    encode_dataset = EncodeDatasetForERNIE(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                           tokenizer, max_len=text_max_length)
    end_time = time.time()
    logger.info(f"tokenization cost: {(end_time - start_time)} s")

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollatorForERNIE(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    # lookup_rep_num = []
    tmp_lookup_pid = []
    tmp_p_rep_num = []
    model = model.to(training_args.device)
    model.eval()

    # load entity embeddings
    """ 推理时使用entity embedding """
    """ 训练时保存的ent_embeddings文件中已经包含了第0行的CLS向量, 不需要再额外添加了 """
    embed = torch.load(model_args.ent_embedding_path, map_location=torch.device('cpu'))
    embed.weight.requires_grad = False
    logger.info(f"ent_embed.weight.requires_grad = {embed.weight.requires_grad}")
    logger.info(f"ent_embed.weight.shape = {embed.weight.shape}")
    logger.info(f"ent_embed.weight.dtype = {embed.weight.dtype}")
    logger.info(f"ent_embed for embed[0]:{embed.weight[0][:10]}")
    """ 推理时使用entity embedding """


    start_time = time.time()
    for (batch_ids, batch) in tqdm(encode_loader):
        if data_args.encode_is_qry:
            lookup_indices.extend(batch_ids)
            # lookup_rep_num.extend(batch["ent_mask"].count_nonzero(dim=1).tolist())
        else:
            tmp_lookup_pid = batch_ids
            tmp_p_rep_num = batch["ent_mask"].count_nonzero(dim=1).tolist()
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                batch["input_ent"] = embed(batch["input_ent"] + 1)
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    # encoded.append(model_output.p_reps.cpu().detach().numpy())
                    p_reps = model_output.p_reps
                    p_reps, pids = prune_passage_representation(p_reps, tmp_p_rep_num, tmp_lookup_pid)
                    encoded.append(p_reps.cpu().detach().numpy())
                    lookup_indices.extend(pids)

    encoded = np.concatenate(encoded)
    end_time = time.time()
    print(f"encoding cost: {(end_time - start_time)} s")

    assert len(lookup_indices) == encoded.shape[0]

    if not os.path.exists(os.path.dirname(data_args.encoded_save_path)):
        os.makedirs(os.path.dirname(data_args.encoded_save_path))

    # encoded: <class 'numpy.ndarray'>, shape: (nums of all single psg reps, dim)   -> passage
    # encoded: <class 'numpy.ndarray'>, shape: (nums of qry reps, dim)   -> query
    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()

    pass
