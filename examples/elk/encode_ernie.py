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
from tevatron.modeling import EncoderOutput, ERNIEModel
from tevatron.datasets import HFQueryDataset, HFCorpusDataset
from knowledge_bert.modeling import BertConfig

logger = logging.getLogger(__name__)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    ernie_tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name,
        do_lower_case=True,  # use a uncased pretrained model
    )
    model = ERNIEModel.load(
        model_name_or_path=model_args.model_name_or_path,
        untie_encoder=model_args.untie_encoder,
        qry_use_ernie=model_args.qry_use_ernie,
        cache_dir=model_args.cache_dir,
    )

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(tokenizer=ernie_tokenizer, data_args=data_args,
                                        cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    else:
        encode_dataset = HFCorpusDataset(tokenizer=ernie_tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    start_time = time.time()
    encode_dataset = EncodeDatasetForERNIE(encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
                                   ernie_tokenizer, max_len=text_max_length)
    end_time = time.time()
    logger.info(f"tokenization cost: {(end_time - start_time)} s")

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollatorForERNIE(
            ernie_tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
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

    """  加载IDF文件  """
    if model_args.idf_file_path:
        tmp = np.load(model_args.idf_file_path)
        tmp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(dim=1)
        idf = torch.nn.Embedding.from_pretrained(tmp, freeze=True)  # dtype=torch.float32
        idf.weight.requires_grad = False
        del tmp
        logger.info(f"idf.weight.requires_grad = {idf.weight.requires_grad}")
        logger.info(f"idf.weight.shape = {idf.weight.shape}")
        logger.info(f"idf.weight.dtype = {idf.weight.dtype}")
        logger.info(f"idf for idf[0]:{idf.weight[0][:10]}")
    else:
        idf = None
    """  加载IDF文件  """

    start_time = time.time()
    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
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
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded)
    end_time = time.time()
    print(f"encoding cost: {(end_time - start_time)} s")

    if not os.path.exists(os.path.dirname(data_args.encoded_save_path)):
        os.makedirs(os.path.dirname(data_args.encoded_save_path))

    # encoded: <class 'numpy.ndarray'>, shape: (nums, dim)
    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()

    pass
