import logging
import os
import sys

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, BertPreTrainedModel
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.data import TrainDatasetForERNIE, QPCollatorForERNIE, TrainDatasetForERNIEWithRE, QPCollatorForERNIEWithRE
from tevatron.modeling import EVASingleModel, EVASingleREModel
from tevatron.trainer import ErnieTrainer
from tevatron.datasets import HFTrainDataset, HFTrainDatasetWithRE
from knowledge_bert.tokenization import BertTokenizer
from typing import List

logger = logging.getLogger(__name__)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    print(f"sys.argv: {sys.argv}")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use "
            f"--overwrite_output_dir to overcome. "
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    # load pretrained entity embeddings && prepend a zero vector for non-entity
    tmp = torch.load(model_args.ent_embedding_path, map_location=torch.device('cpu'))
    zero = torch.tensor([0] * int(model_args.ent_hidden_size)).unsqueeze(dim=0)  # zero embedding for non-entity
    embed = torch.concat((zero, tmp.weight), dim=0)

    # 默认情况下，torch.nn.Embedding.from_pretrained类方法冻结参数。如果要训练参数，则需要将freeze关键字参数设置为False
    embed = torch.nn.Embedding.from_pretrained(embed, freeze=True)  # dtype=torch.float32
    embed.weight.requires_grad = False  # 固定entity embedding
    # embed.weight.requires_grad = True  # 可训练的entity embedding
    logger.info(f"ent_embed.weight.requires_grad = {embed.weight.requires_grad}")
    logger.info(f"ent_embed.weight.shape = {embed.weight.shape}")
    logger.info(f"ent_embed.weight.dtype = {embed.weight.dtype}")
    logger.info(f"ent_embed for embed[0]:{embed.weight[0][:10]}")
    logger.info(f"ent_embed device:{embed.weight.device}")
    del tmp

    """ 加载IDF文件 """
    if model_args.idf_file_path:
        tmp = np.load(model_args.idf_file_path)
        tmp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(dim=1)
        idf = torch.nn.Embedding.from_pretrained(tmp, freeze=True)  # dtype=torch.float32
        idf.weight.requires_grad = False
        logger.info(f"idf.weight.requires_grad = {idf.weight.requires_grad}")
        logger.info(f"idf.weight.shape = {idf.weight.shape}")
        logger.info(f"idf.weight.dtype = {idf.weight.dtype}")
        logger.info(f"idf for idf[0]:{idf.weight[0][:10]}")
        logger.info(f"idf device:{idf.weight.device}")
        del tmp
    else:
        idf = None
    """  加载IDF文件  """

    model_args.num_labels = 1
    # ernie_tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name,
        do_lower_case=True,  # use a uncased pretrained model
    )
    model = EVASingleREModel.build(
        model_args,
        training_args,
        embed=embed,
        idf=idf,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = HFTrainDatasetWithRE(tokenizer=tokenizer, data_args=data_args,
                                         cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = TrainDatasetForERNIEWithRE(data_args, train_dataset.process(), tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    trainer_cls = ErnieTrainer if training_args.grad_cache else ErnieTrainer
    # model_args.ernie_model_path,
    # model_args.ent_hidden_size,
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollatorForERNIEWithRE(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    if not os.path.exists(os.path.dirname(training_args.output_dir)):
        os.makedirs(os.path.dirname(training_args.output_dir))
    args_txt_path = os.path.join(training_args.output_dir, "args.txt")
    with open(args_txt_path, "w")as f:
        f.write("[1]model args:\n" + str(model_args) + "\n\n")
        f.write("[2]data args:\n" + str(data_args) + "\n\n")
        f.write("[3]training args:\n" + str(training_args) + "\n\n")

    trainer.train()
    trainer.save_model()

    # if trainer.is_world_process_zero():
    #     tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
    print(1)


    def function(arr: List[List[int]], S: int) -> List[int]:
        ans = [0] * (S + 1)
        for a, b in arr:
            for i in range(a, b + 1):
                ans[i] += 1
        return ans




