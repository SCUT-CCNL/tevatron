### 异步评估dense retrieval模型, 只对ERNIE/ELK_Multi有效
export CUDA_VISIBLE_DEVICES=1

### retrieve_batch_size是指每次Faiss检索时查询的数量, -1表示全部a

## TREC-COVID
#python -m asyncval.eval_elk_multi \
#  --model_name ELK_Multi \
#	--query_file dataset/trec-covid/scibert_linked_queries.test.tsv \
#	--candidate_dir dataset/trec-covid-knowledge-corpus/knowledge_corpus_constrainTuis_with_types.jsonl \
#	--ent_cluster_dir dataset/trec-covid-cluster/trec_covid_doc_with_ent_cluster_max5.jsonl \
#	--ckpts_dir output/model_covid/ELK_Multi/v5 \
#	--tokenizer_name_or_path pretrained_model/co-condenser-marco \
#	--qrel_file dataset/trec-covid/qrels.test.tsv \
#	--encoded_corpus_save_path precompute/trec-covid/ELK_Multi/v5 \
#  --untie_encoder False \
#  --qry_use_ernie True \
#	--metrics R@20 R@100 R@1000 nDCG@10 nDCG@20 \
#	--per_device_eval_batch_size 64 \
#	--retrieve_batch_size -1 \
#	--ent_hidden_size 100 \
#  --depth 6000 \
#  --fp16 \
#  --q_max_len 12 \
#  --p_max_len 510 \
#  --report_to tensorboard \
#  --dataloader_num_workers 12 \
#  --logging_dir logs/trec-covid/ELK_Multi/v5/debug/test \
#	--output_dir result/trec-covid/ELK_Multi/v5/debug/test


### NFCorpus
# TODO 保存query embs
python -m asyncval.eval_elk_multi \
  --model_name ELK_Multi \
	--query_file dataset/NFCorpus/scibert_linked_queries.test.tsv \
	--candidate_dir dataset/NFCorpus-knowledge-corpus/knowledge_corpus_constrainTuis_with_types.jsonl \
	--ent_cluster_dir dataset/NFCorpus-cluster/nfcorpus_doc_with_ent_cluster_max5.jsonl \
	--ckpts_dir output/model_nfc/ELK_Multi/v4 \
	--tokenizer_name_or_path pretrained_model/co-condenser-marco \
	--qrel_file dataset/NFCorpus/qrels.test.tsv \
  --encoded_corpus_save_path precompute/NFCorpus/ELK_Multi/v4/max5 \
  --untie_encoder False \
  --qry_use_ernie True \
	--metrics R@20 R@100 R@1000 nDCG@10 nDCG@20\
	--per_device_eval_batch_size 64 \
	--retrieve_batch_size -1 \
	--ent_hidden_size 100 \
  --depth 6000 \
  --fp16 \
  --q_max_len 12 \
  --p_max_len 510 \
  --report_to tensorboard \
  --dataloader_num_workers 10 \
  --logging_dir logs/NFCorpus/ELK_Multi/v4/debug/test \
	--output_dir result/NFCorpus/ELK_Multi/v4/debug/test



### TREC-COVID
## ELK_Multi + 其他多向量方法:
#python -m asyncval.eval_elk_multi_eva_multi \
#  --model_name ELK_Multi_EVA_Multi \
#	--query_file dataset/trec-covid/scibert_linked_queries.test.tsv \
#	--candidate_dir dataset/trec-covid-knowledge-corpus/knowledge_corpus_constrainTuis_with_types.jsonl \
#	--eva_cluster_dir dataset/trec-covid-cluster/trec_covid_eva_cluster_for_doc.jsonl \
#	--ckpts_dir output/model_covid/ELK_Multi_EVA_Multi \
#	--tokenizer_name_or_path pretrained_model/co-condenser-marco \
#	--qrel_file dataset/trec-covid/qrels.test.tsv \
#	--encoded_corpus_save_path precompute/trec-covid/ELK_Multi_EVA_Multi \
#  --untie_encoder False \
#  --qry_use_ernie True \
#	--metrics R@20 R@100 R@1000 nDCG@10 nDCG@20 \
#	--per_device_eval_batch_size 64 \
#	--retrieve_batch_size -1 \
#	--ent_hidden_size 100 \
#  --depth 20000 \
#  --fp16 \
#  --q_max_len 12 \
#  --p_max_len 510 \
#  --report_to tensorboard \
#  --dataloader_num_workers 8 \
#  --logging_dir logs/trec-covid/ELK_Multi_EVA_Multi/test \
#	--output_dir result/trec-covid/ELK_Multi_EVA_Multi/test


### NFCorpus
## ELK_Multi + 其他多向量方法:
#python -m asyncval.eval_elk_multi_eva_multi \
#  --model_name ELK_Multi_EVA_Multi \
#	--query_file dataset/NFCorpus/scibert_linked_queries.test.tsv \
#	--candidate_dir dataset/NFCorpus-knowledge-corpus/knowledge_corpus_constrainTuis_with_types.jsonl \
#	--eva_cluster_dir dataset/NFCorpus-cluster/nfcorpus_eva_cluster_for_doc.jsonl \
#	--ckpts_dir output/model_nfc/ELK_Multi_EVA_Multi \
#	--tokenizer_name_or_path pretrained_model/co-condenser-marco \
#	--qrel_file dataset/NFCorpus/qrels.test.tsv \
#  --encoded_corpus_save_path precompute/NFCorpus/ELK_Multi_EVA_Multi/ \
#  --untie_encoder False \
#  --qry_use_ernie True \
#	--metrics R@20 R@100 R@1000 nDCG@10 nDCG@20\
#	--per_device_eval_batch_size 64 \
#	--retrieve_batch_size -1 \
#	--ent_hidden_size 100 \
#  --depth 20000 \
#  --fp16 \
#  --q_max_len 12 \
#  --p_max_len 510 \
#  --report_to tensorboard \
#  --dataloader_num_workers 8 \
#  --logging_dir logs/NFCorpus/ELK_Multi_EVA_Multi/test \
#	--output_dir result/NFCorpus/ELK_Multi_EVA_Multi/test





