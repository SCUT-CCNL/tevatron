from typing import Dict
from argparse import ArgumentParser
import pytrec_eval


def get_metric(qrels: str, trec: str, metric: str = 'ndcg_cut_10') -> Dict[str, float]:
    with open(qrels, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    with open(trec, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    for query_id, query_measures in sorted(results.items()):
        pass
    mes = {}
    for measure in sorted(query_measures.keys()):
        mes[measure] = pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in
                                                                        results.values()])
    if type(metric) is list:
        return mes
    else:
        return mes[metric]


def get_mrr(qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
    k = int(metric.split('_')[-1])

    qrel = {}
    with open(qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(trec, 'r') as f_run:
        for line in f_run:
            qid, _, did, _, _, _ = line.strip().split()
            if qid not in run:
                run[qid] = []
            run[qid].append(did)

    mrr = 0.0
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i + 1)
                break
        mrr += rr
    mrr /= len(run)
    return mrr


def get_metric_in_topic_level(qrels: str, trec: str, metric: str = 'ndcg_cut_10') -> Dict[str, float]:
    with open(qrels, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    with open(trec, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    res = []
    for query_id, query_measures in sorted(results.items()):
        query_measures["qid"] = query_id
        tmp = {"qid": query_id, "ndcg_cut_10": round(query_measures["ndcg_cut_10"], 4)}
        res.append(tmp)
    print(res)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--qrels', type=str, required=True)
    parser.add_argument('--trec', type=str, required=True)
    args = parser.parse_args()

    metrics = ["recall_10", "recall_20", "recall_100", "recall_500", "recall_1000", "ndcg_cut_10", "ndcg_cut_20"]
    mrr_10 = get_mrr(args.qrels, args.trec, "mrr_10")
    mes = get_metric(args.qrels, args.trec, metrics)
    print(f'MRR@10: {mrr_10}')
    for metric in metrics:
        print(f'{metric}: {mes[metric]}')

    # get_metric_in_topic_level(args.qrels, args.trec, metrics)
    pass
