from transformers import PreTrainedTokenizer
from typing import List, Dict

from knowledge_bert.tokenization import BertTokenizer


def create_features(tokenizer, text: str, ents: list, max_length: int, is_query=False,
                    psg_ent_mapping: List[list] = None, ent_cluster_masks: List[list] = None):
    """
    tokenize -> truncate -> add special tokens -> convert to ids (not padding here)
    -> get attention mask -> do padding
    """
    # 1. tokenize
    tokens, ents = tokenizer.tokenize(text, ents)
    if psg_ent_mapping is not None and len(psg_ent_mapping) != 0:
        assert len(tokens) == len(psg_ent_mapping[0])
    if ent_cluster_masks is not None and len(ent_cluster_masks) != 0:
        assert len(tokens) == len(ent_cluster_masks[0])

    # 2. truncate
    if len(tokens) > max_length - 2:
        tokens = tokens[:(max_length - 2)]
        ents = ents[:(max_length - 2)]
        if psg_ent_mapping is not None:
            for i, ent_mapping in enumerate(psg_ent_mapping):
                psg_ent_mapping[i] = ent_mapping[: max_length - 2]
        if ent_cluster_masks is not None:
            for i, cluster_mask in enumerate(ent_cluster_masks):
                ent_cluster_masks[i] = cluster_mask[: max_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    ents = ["UNK"] + ents + ["UNK"]

    # 3. convert tokens to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ent = []
    ent_mask = []
    for ent in ents:
        if ent != "UNK":
            input_ent.append(int(ent))
            ent_mask.append(1)
        else:
            input_ent.append(-1)
            ent_mask.append(0)
    if not is_query:
        ent_mask[0] = 1

    # 4. get attention mask
    attention_mask = [1] * len(input_ids)
    assert len(input_ids) == len(attention_mask) == len(input_ent) == len(ent_mask)

    # 5. do padding
    padding = [0] * (max_length - len(input_ids))
    padding_2 = [-1] * (max_length - len(input_ids))
    input_ids += padding
    attention_mask += padding
    input_ent += padding_2
    ent_mask += padding
    if psg_ent_mapping is not None:
        for i, ent_mapping in enumerate(psg_ent_mapping):
            psg_ent_mapping[i] = [0] + ent_mapping + [0] + padding
            assert len(input_ids) == len(psg_ent_mapping[i])
    if ent_cluster_masks is not None:
        for i, cluster_mask in enumerate(ent_cluster_masks):
            ent_cluster_masks[i] = [0] + cluster_mask + [0] + padding
            assert len(input_ids) == len(ent_cluster_masks[i])

    if psg_ent_mapping is not None:
        return input_ids, attention_mask, input_ent, ent_mask, psg_ent_mapping
    if ent_cluster_masks is not None:
        return input_ids, attention_mask, input_ent, ent_mask, ent_cluster_masks
    return input_ids, attention_mask, input_ent, ent_mask


def create_features_eva(tokenizer, text: str, ents: list, max_length: int, psg_ent_mapping: List[list] = None):
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
    if psg_ent_mapping is not None:
        for i, emapping in enumerate(psg_ent_mapping):
            psg_ent_mapping[i] = emapping[: max_length - 2]

    attention_mask = [1] * len(input_ids)
    ent_len = len(ents) - ents.count(-1)
    ent_mask = [1] * ent_len + [0] * (len(ents) - ent_len)

    padding = [0] * (max_length - len(input_ids))
    input_ids += padding
    attention_mask += padding

    if psg_ent_mapping is not None:
        for i, emapping in enumerate(psg_ent_mapping):
            psg_ent_mapping[i] = [0] + emapping + [0] + padding
            assert len(input_ids) == len(psg_ent_mapping[i])

    if psg_ent_mapping is not None:
        return input_ids, attention_mask, ents, ent_mask, psg_ent_mapping
    return input_ids, attention_mask, ents, ent_mask


class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        """
        example: {
            "query_id":xxx,
            "query":xxx,
            "positive_passages":["docid":xxx, "title":xxx, "text":xxx],
            "negative_passages":["docid":xxx, "title":xxx, "text":xxx],
        }
        """
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'text_id': query_id, 'text': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        return {'text_id': docid, 'text': text}


class TrainPreProcessorForELK:
    def __init__(self, tokenizer, query_max_length=12, text_max_length=400, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        """
        input:
            example: {
                "query_id":xxx,
                "query":xxx,
                "positive_passages":["docid":xxx, "title":xxx, "text":xxx, "ents": [[triple1], [triple2], ...]],
                "negative_passages":["docid":xxx, "title":xxx, "text":xxx], "ents": [[triple1], [triple2], ...]],
            }
        return:
            1.query: [query tokens ids, query att masks, query ent ids, query ent masks]

            2.positives: [[psg1 token ids, psg1 att masks, psg1 ent ids, psg1 ent masks],
                        [psg2 token ids, psg2 att masks, psg2 ent ids, psg2 ent masks], ...]
                where:
                ent ids: [-1,...,entity id,...-1]
                ent masks: [0,...,1,...,0]
            3.negatives: ...
        """
        qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks = create_features(self.tokenizer,
                                                                                            example["query"],
                                                                                            example["query_ents"],
                                                                                            self.query_max_length,
                                                                                            is_query=True)
        query = [qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks]

        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks = create_features(self.tokenizer,
                                                                                                text,
                                                                                                pos["ents"],
                                                                                                self.text_max_length,
                                                                                                is_query=False)
            positives.append([psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks])

        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks = create_features(self.tokenizer,
                                                                                                text,
                                                                                                neg["ents"],
                                                                                                self.text_max_length,
                                                                                                is_query=False)
            negatives.append([psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks])

        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessorForELK:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks = create_features(self.tokenizer,
                                                                                            example["query"],
                                                                                            example["query_ents"],
                                                                                            self.query_max_length,
                                                                                            is_query=True)
        return {'text_id': query_id, 'text': [qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks]}


class CorpusPreProcessorForELK:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks = create_features(self.tokenizer,
                                                                                            text,
                                                                                            example["ents"],
                                                                                            self.text_max_length,
                                                                                            is_query=False)
        return {'text_id': docid, 'text': [psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks]}


class TrainPreProcessorForELKMulti:
    def __init__(self, cluster_dataset_dict: Dict[int, dict], tokenizer: BertTokenizer, query_max_length=12,
                 text_max_length=400, separator=' '):
        self.cluster_dataset_dict = cluster_dataset_dict
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        """
        input:
            example: {
                "query_id":xxx,
                "query":xxx,
                "positive_passages":["docid":xxx, "title":xxx, "text":xxx, "ents": [[triple1], [triple2], ...]],
                "negative_passages":["docid":xxx, "title":xxx, "text":xxx], "ents": [[triple1], [triple2], ...]],
            }
        return:
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
            3.negatives: ...
        """
        qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks = create_features(self.tokenizer,
                                                                                            example["query"],
                                                                                            example["query_ents"],
                                                                                            self.query_max_length,
                                                                                            is_query=True)
        query = [qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks]

        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            ent_cluster_data = self.cluster_dataset_dict[int(pos['docid'])]
            ent_cluster_data = ent_cluster_data['ent_cluster_masks'].copy()

            psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks, psg_ent_cluster_masks = \
                create_features(self.tokenizer, text, pos["ents"], self.text_max_length, False, psg_ent_mapping=None,
                                ent_cluster_masks=ent_cluster_data)

            positives.append([[psg_input_ids], [psg_attention_masks], [psg_input_ents], [psg_ent_masks],
                              psg_ent_cluster_masks])

        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            ent_cluster_data = self.cluster_dataset_dict[int(neg['docid'])]
            ent_cluster_data = ent_cluster_data['ent_cluster_masks'].copy()

            psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks, psg_ent_cluster_masks = \
                create_features(self.tokenizer, text, neg["ents"], self.text_max_length, False, psg_ent_mapping=None,
                                ent_cluster_masks=ent_cluster_data)

            negatives.append([[psg_input_ids], [psg_attention_masks], [psg_input_ents], [psg_ent_masks],
                              psg_ent_cluster_masks])

        return {'query': query, 'positives': positives, 'negatives': negatives}


class QueryPreProcessorForELKMulti:
    """
    same with QueryPreProcessorForELK
    """
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks = create_features(self.tokenizer,
                                                                                            example["query"],
                                                                                            example["query_ents"],
                                                                                            self.query_max_length,
                                                                                            is_query=True)
        return {'text_id': query_id, 'text': [qry_input_ids, qry_attention_masks, qry_input_ents, qry_ent_masks]}


class CorpusPreProcessorForELKMulti:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']

        psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks, psg_ent_cluster_masks = \
            create_features(self.tokenizer, text, example["ents"], self.text_max_length, is_query=False,
                            psg_ent_mapping=None, ent_cluster_masks=example["ent_cluster_masks"])
        return {'text_id': docid, 'text': [psg_input_ids, psg_attention_masks, psg_input_ents, psg_ent_masks]}
