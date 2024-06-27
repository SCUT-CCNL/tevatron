import random
from dataclasses import dataclass
from typing import List, Tuple, Union

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        """
        return:
            q_collated <BatchEncoding>: {
                "input_ids": tensor [bs, qry_max_len],
                "attention_mask": tensor [bs, qry_max_len],
            }
            p_collated <BatchEncoding>: {
                "input_ids": tensor [bs * train_n_passage, psg_max_len],
                "attention_mask": tensor [bs * train_n_passage, psg_max_len],
            }
        """
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features


class TrainDatasetForELK(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        group:
            1.query: [query tokens ids, query ent ids, query ent masks]
            2.positives: [[psg1 token ids, psg1 ent ids, psg1 ent masks], [psg2 token ids, psg2 ent ids, psg2 ent masks], ...]
                ent ids: [-1,...,entity id,...-1]
                ent masks: [0,...,1,...,0]
            3.negatives: ...
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = BatchEncoding({"input_ids": qry[0], "attention_mask": qry[1],
                                       "input_ent": qry[2], "ent_mask": qry[3]})

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(BatchEncoding({"input_ids": pos_psg[0], "attention_mask": pos_psg[1],
                                               "input_ent": pos_psg[2], "ent_mask": pos_psg[3]}))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(BatchEncoding({"input_ids": neg_psg[0], "attention_mask": neg_psg[1],
                                                   "input_ent": neg_psg[2], "ent_mask": neg_psg[3]}))

        return encoded_query, encoded_passages


class EncodeDatasetForELK(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        """
        self.encode_data[item]:
            1. 'text_id': docid or query_id
            2. 'text': [input_ids, attention_masks, input_ents, ent_masks]}
        """
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = BatchEncoding({"input_ids": text[0], "attention_mask": text[1],
                                      "input_ent": text[2], "ent_mask": text[3]})
        return text_id, encoded_text


@dataclass
class QPCollatorForELK(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        """
        do padding here and convert list to tensor to construct batch data
        return:
            q_collated: {
                "input_ids": tensor [bs, qry_max_len],
                "attention_mask": tensor [bs, qry_max_len],
                "input_ent": tensor [bs, qry_max_len],
                "ent_mask": tensor [bs, qry_max_len],
            }
            p_collated: {
                "input_ids": tensor [bs * train_n_passage, psg_max_len],
                "attention_mask": tensor [bs * train_n_passage, psg_max_len],
                "input_ent": tensor [bs * train_n_passage, psg_max_len],
                "ent_mask": tensor [bs * train_n_passage, psg_max_len],
            }
        """
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = {key: torch.tensor([example[key] for example in qq]) for key in qq[0].keys()}
        d_collated = {key: torch.tensor([example[key] for example in dd]) for key in dd[0].keys()}
        return q_collated, d_collated


@dataclass
class EncodeCollatorForELK(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        if isinstance(text_features[0], list):
            text_features = sum(text_features, [])
        collated_features = {key: torch.tensor([example[key] for example in text_features]) for key in
                             text_features[0].keys()}
        return text_ids, collated_features


class TrainDatasetWithRE(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        group:
            1.query: [query tokens ids, query attention mask]
            2.positives:
            [
                [psg1 token ids, psg1 atten mask,psg1 h_ent_ids, psg1 t_ent_ids, psg1 RE labels, psg1 ent_mapping],
                [psg2 token ids, psg2 atten mask, psg2 h_ent_ids, psg2 t_ent_ids, psg2 RE labels, psg2 ent_mapping],
                ...
            ]
            3.negatives: ...

        return:
            encoded_query: [query tokens ids, query attention mask]  -> BatchEncoding
            encoded_passages: [
                [psg1 token ids, psg1 atten mask,psg1 h_ent_ids, psg1 t_ent_ids, psg1 RE labels, psg1 ent_mapping],
                ...
            ]   -> List[BatchEncoding]]
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = BatchEncoding({"input_ids": qry[0], "attention_mask": qry[1]})

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        encoded_passages.append(BatchEncoding({"input_ids": pos_psg[0][0], "attention_mask": pos_psg[1][0],
                                               "h_ent_ids": pos_psg[2][0], "t_ent_ids": pos_psg[3][0],
                                               "labels": pos_psg[4][0], "ent_mapping": pos_psg[5]}))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(BatchEncoding({"input_ids": neg_psg[0][0], "attention_mask": neg_psg[1][0],
                                                   "h_ent_ids": neg_psg[2][0], "t_ent_ids": neg_psg[3][0],
                                                   "labels": neg_psg[4][0], "ent_mapping": neg_psg[5]}))

        return encoded_query, encoded_passages


class TrainDatasetForELKWithRE(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        group:
            1.query: [query tokens ids, query att masks, query ent ids, query ent masks]
            2.positives:
            [
                [psg1 token ids, psg1 att masks, psg1 ent ids, psg1 ent masks, psg1 h_ent_ids, psg1 t_ent_ids,
                psg1 RE labels, psg1 ent_mapping],
                [psg2 token ids, psg2 att masks, psg2 ent ids, psg2 ent masks, psg2 h_ent_ids, psg2 t_ent_ids,
                psg2 RE labels, psg2 ent_mapping],
                ...
            ]
                where:
                ent ids: [-1,...,entity id,...-1]
                ent masks: [0,...,1,...,0]
            3.negatives: ...

        return:
            encoded_query: [query tokens ids, query attention mask, query ent ids, query ent masks]  -> BatchEncoding
            encoded_passages: [
                [psg1 token ids, psg1 atten mask, psg1 ent ids, psg ent mask, psg1 h_ent_ids, psg1 t_ent_ids,
                psg1 RE labels, psg1 ent_mapping],
                ...
            ]   -> List[BatchEncoding]]
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = BatchEncoding({"input_ids": qry[0], "attention_mask": qry[1],
                                       "input_ent": qry[2], "ent_mask": qry[3]})

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        encoded_passages.append(BatchEncoding({"input_ids": pos_psg[0][0], "attention_mask": pos_psg[1][0],
                                               "input_ent": pos_psg[2][0], "ent_mask": pos_psg[3][0],
                                               "h_ent_ids": pos_psg[4][0], "t_ent_ids": pos_psg[5][0],
                                               "labels": pos_psg[6][0], "ent_mapping": pos_psg[7]}))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(BatchEncoding({"input_ids": neg_psg[0][0], "attention_mask": neg_psg[1][0],
                                                   "input_ent": neg_psg[2][0], "ent_mask": neg_psg[3][0],
                                                   "h_ent_ids": neg_psg[4][0], "t_ent_ids": neg_psg[5][0],
                                                   "labels": neg_psg[6][0], "ent_mapping": neg_psg[7]}))

        return encoded_query, encoded_passages


@dataclass
class QPCollatorForELKWithRE(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def padding_in_batch_entity_data(self, batch_encoded_passages: List) -> List[BatchEncoding]:
        max_entity_num = -1
        max_REpair_num = -1
        max_doc_text_length = len(batch_encoded_passages[0]['input_ids'])
        for i, doc_record in enumerate(batch_encoded_passages):
            max_entity_num = max(len(doc_record['ent_mapping']), max_entity_num)
            max_REpair_num = max(len(doc_record['labels']), max_REpair_num)

        if max_entity_num == 0:
            max_entity_num = 1
        if max_REpair_num == 0:
            max_REpair_num = 1

        entity_mapping_padding = [0] * max_doc_text_length
        for i, doc_record in enumerate(batch_encoded_passages):
            REpair_num_diff = max_REpair_num - len(doc_record['h_ent_ids'])
            entity_num_diff = max_entity_num - len(doc_record['ent_mapping'])
            batch_encoded_passages[i]['label_mask'] = [1] * len(doc_record['labels']) + [0] * REpair_num_diff
            batch_encoded_passages[i]['ent_mapping_mask'] = [1] * len(doc_record['ent_mapping']) + [0] * entity_num_diff

            batch_encoded_passages[i]['h_ent_ids'] = doc_record['h_ent_ids'] + [0] * REpair_num_diff
            batch_encoded_passages[i]['t_ent_ids'] = doc_record['t_ent_ids'] + [0] * REpair_num_diff
            batch_encoded_passages[i]['labels'] = doc_record['labels'] + [0] * REpair_num_diff
            batch_encoded_passages[i]['ent_mapping'] = doc_record['ent_mapping'] + [entity_mapping_padding for i in
                                                                                    range(entity_num_diff)]

        return batch_encoded_passages

    def __call__(self, features):
        """
        do padding here and convert list to tensor to construct batch data
        return:
            q_collated: {
                "input_ids": tensor [bs, qry_max_len],
                "attention_mask": tensor [bs, qry_max_len],
                "input_ent": tensor [bs, qry_max_len],  (optional)
                "ent_mask": tensor [bs, qry_max_len],   (optional)
            }
            p_collated: {
                "input_ids": tensor [bs * train_n_passage, psg_max_len],
                "attention_mask": tensor [bs * train_n_passage, psg_max_len],
                "input_ent": tensor [bs * train_n_passage, psg_max_len],    (optional)
                "ent_mask": tensor [bs * train_n_passage, psg_max_len],     (optional)
                "h_ent_ids":    (required)
                "t_ent_ids":    (required)
                "labels":       (required)
                "ent_mapping":  (required)
            }
        """
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        dd = self.padding_in_batch_entity_data(dd)
        q_collated = {key: torch.tensor([example[key] for example in qq]) for key in qq[0].keys()}
        d_collated = {key: torch.tensor([example[key] for example in dd]) for key in dd[0].keys()}
        return q_collated, d_collated


class TrainDatasetForELKMulti(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        group:
            1.query: [query tokens ids, query att masks, query ent ids, query ent masks]
            2.positives:
            [
                [psg1 token ids, psg1 att masks, psg1 ent ids, psg1 ent masks, psg1 ent cluster masks],
                [psg2 token ids, psg2 att masks, psg2 ent ids, psg2 ent masks, psg2 ent cluster masks],
                ...
            ]
                where:
                ent ids: [-1,...,entity id,...-1]
                ent masks: [0,...,1,...,0]
            3.negatives:

        return:
            encoded_query: [query tokens ids, query attention mask, query ent ids, query ent masks]  -> BatchEncoding
            encoded_passages: [
                [psg1 token ids, psg1 atten mask, psg1 ent ids, psg1 ent mask, psg1 ent cluster masks],
                ...
            ]   -> List[BatchEncoding]]
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = BatchEncoding({"input_ids": qry[0], "attention_mask": qry[1],
                                       "input_ent": qry[2], "ent_mask": qry[3]})

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        encoded_passages.append(BatchEncoding({"input_ids": pos_psg[0][0], "attention_mask": pos_psg[1][0],
                                               "input_ent": pos_psg[2][0], "ent_mask": pos_psg[3][0],
                                               "ent_cluster_masks": pos_psg[4]}))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(BatchEncoding({"input_ids": neg_psg[0][0], "attention_mask": neg_psg[1][0],
                                                   "input_ent": neg_psg[2][0], "ent_mask": neg_psg[3][0],
                                                   "ent_cluster_masks": neg_psg[4]}))

        return encoded_query, encoded_passages


@dataclass
class QPCollatorForELKMulti(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        """
        do padding here and convert list to tensor to construct batch data
        return:
            q_collated: {
                "input_ids": tensor [bs, qry_max_len],
                "attention_mask": tensor [bs, qry_max_len],
                "input_ent": tensor [bs, qry_max_len],  (optional)
                "ent_mask": tensor [bs, qry_max_len],   (optional)
            }
            p_collated: {
                "input_ids": tensor [bs * train_n_passage, psg_max_len],
                "attention_mask": tensor [bs * train_n_passage, psg_max_len],
                "input_ent": tensor [bs * train_n_passage, psg_max_len],    (optional)
                "ent_mask": tensor [bs * train_n_passage, psg_max_len],     (optional)
                "ent_cluster_masks":    (required)
            }
        """
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated = {key: torch.tensor([example[key] for example in qq]) for key in qq[0].keys()}
        d_collated = {key: torch.tensor([example[key] for example in dd]) for key in dd[0].keys()}
        return q_collated, d_collated


class TrainDatasetForELKMultiEVAMulti(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        group:
            1.query: [query tokens ids, query att masks, query ent ids, query ent masks]
            2.positives:
            [
                [psg1 token ids, psg1 att masks, psg1 ent ids, psg1 ent masks, psg1 eva cluster ids],
                [psg2 token ids, psg2 att masks, psg2 ent ids, psg2 ent masks, psg2 eva cluster ids],
                ...
            ]
                where:
                ent ids: [-1,...,entity id,...-1]
                ent masks: [0,...,1,...,0]
            3.negatives:

        return:
            encoded_query: [query tokens ids, query attention mask, query ent ids, query ent masks]  -> BatchEncoding
            encoded_passages: [
                [psg1 token ids, psg1 atten mask, psg1 ent ids, psg1 ent mask, psg1 eva cluster ids],
                ...
            ]   -> List[BatchEncoding]]
        """
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = BatchEncoding({"input_ids": qry[0], "attention_mask": qry[1],
                                       "input_ent": qry[2], "ent_mask": qry[3]})

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        encoded_passages.append(BatchEncoding({"input_ids": pos_psg[0][0], "attention_mask": pos_psg[1][0],
                                               "input_ent": pos_psg[2][0], "ent_mask": pos_psg[3][0],
                                               "input_cluster": pos_psg[4][0]}))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(BatchEncoding({"input_ids": neg_psg[0][0], "attention_mask": neg_psg[1][0],
                                                   "input_ent": neg_psg[2][0], "ent_mask": neg_psg[3][0],
                                                   "input_cluster": neg_psg[4][0]}))

        return encoded_query, encoded_passages
