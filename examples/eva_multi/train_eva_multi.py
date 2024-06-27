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
from tevatron.data import TrainDatasetForERNIE, QPCollatorForERNIE
from tevatron.modeling import EVAMultiModel
from tevatron.trainer import ErnieTrainer
from tevatron.datasets import HFTrainDataset

logger = logging.getLogger(__name__)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    model_args.num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = EVAMultiModel.build(
        model_args,
        training_args,
        embed=embed,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = TrainDatasetForERNIE(data_args, train_dataset.process(), tokenizer)
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
        data_collator=QPCollatorForERNIE(
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

    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
    from transformers.modeling_outputs import BaseModelOutput

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # tmp = torch.load("/data1/home/tanjiajie/model/tevatron/ernie_base/kg_embedding/ent_embeddings_covid19_100d.pt",
    #                  map_location=torch.device('cpu'))
    # # tmp_t = tmp.weight.clone().detach().cpu()
    # zero = torch.tensor([0] * 100).unsqueeze(dim=0)  # zero embedding for non-entity
    # embed = torch.concat((zero, tmp.weight), dim=0)
    # embed = torch.nn.Embedding.from_pretrained(embed)
    # embed.weight.requires_grad = False
    # del tmp
    # a = torch.tensor([0, 1, 2])
    # a = embed(a)
    print(1)

    # print(12345)
    # a = ["aaa", "bbb"]
    # a = ["[CLS]"] + a + ["[SEP]"]
    # print(a)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # ernie_tokenizer = BertTokenizer.from_pretrained("/data1/home/tanjiajie/model/tevatron/ernie_base")
    # example = {"docid": "138992",
    #            "title": "Monophyletic relationship between severe acute respiratory syndrome coronavirus and group 2 "
    #                     "coronaviruses. Although primary genomic analysis has revealed that severe acute respiratory "
    #                     "syndrome coronavirus (SARS CoV) is a new type of coronavirus, the different protein trees "
    #                     "published in previous reports have provided no conclusive evidence indicating the "
    #                     "phylogenetic position of SARS CoV. To clarify the phylogenetic relationship between SARS CoV "
    #                     "and other coronaviruses, we compiled a large data set composed of 7 concatenated protein "
    #                     "sequences and performed comprehensive analyses, using the maximum-likelihood, "
    #                     "Bayesian-inference, and maximum-parsimony methods. All resulting phylogenetic trees "
    #                     "displayed an identical topology and supported the hypothesis that the relationship between "
    #                     "SARS CoV and group 2 CoVs is monophyletic. Relationships among all major groups were well "
    #                     "resolved and were supported by all statistical analyses.",
    #            "text": "", "ents": [["6", 34, 79], ["6", 92, 105], ["6", 159, 204], ["6", 206, 214], ["10", 233, 244],
    #                                 ["6", 381, 389], ["6", 440, 448], ["6", 459, 472], ["6", 791, 799]]}
    # text = example["title"]
    # ents = example["ents"]
    # tokens, ent_masks = ernie_tokenizer.tokenize(text, ents)
    # tokens_1, ent_masks_1 = ernie_tokenizer.tokenize(text, [])  # No entity here, it's ok
    # text = "what a beautiful city, Litang"
    # tokens_vanilla = tokenizer.encode(text,
    #                                   add_special_tokens=False,
    #                                   max_length=20,
    #                                   truncation=True)
    # print(tokens_vanilla)
    # item = tokenizer.prepare_for_model(
    #     tokens_vanilla,
    #     truncation='only_first',
    #     max_length=20,
    #     padding=True,
    #     return_attention_mask=True,
    #     return_token_type_ids=True,
    # )
    # print(item)
    #
    # ids = ernie_tokenizer.convert_tokens_to_ids(tokens)
    # ids_vanilla = ernie_tokenizer.convert_tokens_to_ids(tokens_1)
    # for ID, ID_van in zip(ids, ids_vanilla):
    #     if ID != ID_van:
    #         raise KeyboardInterrupt
    #
    #
    # print(len(tokens_1), len(tokens))
    # print(len(ids), len(ids_vanilla))

    # pass
