import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    elk_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained ELK model in a local path to build psg encoder. If "
                          "untie_encoder=true, the query encoder will build from 'model_name_or_path'. If "
                          "untie_encoder=false, both encoders will build from 'elk_model_path'."
                          "ONLY USED IN TRAINING!!!"}
    )
    ent_embedding_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained/trained entity embedding..."}
    )
    idf_file_path:  str = field(
        default=None,
        metadata={"help": "Path to idf file for entity/concept..."}
    )
    ent_hidden_size: int = field(
        default=100,
        metadata={"help": "hidden size of entity embedding, remember to set this value together with that in "
                          "elk_config.json!!!"}
    )
    qry_use_elk: bool = field(
        default=False,
        metadata={"help": "Whether the query encoder also uses the ELK model? Set it to True by just using this arg"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders."}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    re_dir: str = field(
        default=None, metadata={"help": "Path to the file of document RE Datasets"}
    )
    ent_cluster_dir: str = field(
        default=None, metadata={"help": "Path to the file of document ent cluster masking Datasets"}
    )
    eva_cluster_dir: str = field(
        default=None, metadata={"help": "Path to the file of document clusters Datasets for EVA-Multi "}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=8, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f)
                    for f in files
                    if f.endswith('jsonl') or f.endswith('json')
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    temperature: Optional[float] = field(default=1.0, metadata={"help": "temperature for softmax when training"})
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    comment: Optional[str] = field(
        default=None, metadata={"help": "description of the training detail"}
    )
