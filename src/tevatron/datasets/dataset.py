from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from transformers import PreTrainedTokenizer
from .preprocessor import TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor, \
TrainPreProcessorForELK, QueryPreProcessorForELK, CorpusPreProcessorForELK, \
    TrainPreProcessorForELKMulti, QueryPreProcessorForELKMulti, CorpusPreProcessorForELKMulti, \
from ..arguments import DataArguments

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
ELK_PROCESSORS = [TrainPreProcessorForELK, QueryPreProcessorForELK, CorpusPreProcessorForELK]
ELK_MULTI_PROCESSORS = [TrainPreProcessorForELKMulti, QueryPreProcessorForELKMulti,
                          CorpusPreProcessorForELKMulti]


PROCESSOR_INFO = {
    'Tevatron/msmarco-passage-elk': ELK_PROCESSORS,
    'Tevatron/msmarco-passage-elk-multi': ELK_MULTI_PROCESSORS,
    # 'json': [None, None, None]
}


class HFTrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[0]

        if data_args.train_dir:
            data_args.dataset_name = "json"
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][1] if data_args.dataset_name in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[1]

        if data_args.encode_in_path:
            data_args.dataset_name = "json"
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization for query...",
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        script_prefix = data_args.dataset_name
        if script_prefix.endswith('-corpus'):
            script_prefix = script_prefix[:-7]
        self.preprocessor = PROCESSOR_INFO[script_prefix][2] if script_prefix in PROCESSOR_INFO \
            else DEFAULT_PROCESSORS[2]

        if data_args.encode_in_path:
            data_args.dataset_name = "json"
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization for docs...",
            )
        return self.dataset


class HFTrainDatasetWithCluster:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        self.cluster_dataset_dict = {}
        data_files = data_args.train_path
        cluster_data_files = [data_args.ent_cluster_dir]
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        if cluster_data_files:
            cluster_data_files = {data_args.dataset_split: cluster_data_files}
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.preprocessor = PROCESSOR_INFO[data_args.dataset_name][0] if data_args.dataset_name in PROCESSOR_INFO \
            else None
        # self.preprocessor = TrainPreProcessorForELKMulti

        if data_args.train_dir:
            data_args.dataset_name = "json"
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir)[data_args.dataset_split]
        self.cluster_dataset = load_dataset(data_args.dataset_name,
                                            data_args.dataset_language,
                                            data_files=cluster_data_files, cache_dir=cache_dir)[data_args.dataset_split]
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        self.cluster_dataset = self.cluster_dataset.shard(shard_num, shard_idx)  # arrow_dataset
        for line in self.cluster_dataset:
            docid = int(line["docid"])
            self.cluster_dataset_dict[docid] = line

        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.cluster_dataset_dict, self.tokenizer, self.q_max_len, self.p_max_len,
                                  self.separator),
                batched=False,
                # num_proc=self.proc_num,
                num_proc=1,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset

