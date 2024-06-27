### 异步评估dense retrieval模型, 只对ERNIE/ERNIE_PubMedBERT有效
export CUDA_VISIBLE_DEVICES=0

### retrieve_batch_size是指每次Faiss检索时查询的数量, -1表示全部a

### TREC-COVID
#python -m asyncval.eval_elk \
#  --model_name ERNIE_PubMedBERT \
#	--query_file dataset/trec-covid/scibert_linked_queries.test.tsv \
#	--candidate_dir dataset/trec-covid-knowledge-corpus/knowledge_corpus_constrainTuis.jsonl \
#	--ckpts_dir output/model_covid/ERNIE_PubMedBERT \
#	--tokenizer_name_or_path pretrained_model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
#	--qrel_file dataset/trec-covid/qrels.test.tsv \
#	--encoded_corpus_save_path precompute/trec-covid/ERNIE_PubMedBERT \
#  --untie_encoder False \
#  --qry_use_ernie True \
#	--metrics R@1000 nDCG@10 nDCG@20 R@20 R@100\
#	--per_device_eval_batch_size 64 \
#	--retrieve_batch_size -1 \
#	--ent_hidden_size 100 \
#  --depth 1000 \
#  --fp16 \
#  --q_max_len 12 \
#  --p_max_len 510 \
#  --report_to tensorboard \
#  --dataloader_num_workers 10 \
#  --logging_dir logs/trec-covid/ERNIE_PubMedBERT/entityWeight-0.1/test \
#	--output_dir result/trec-covid/ERNIE_PubMedBERT/entityWeight-0.1/test



### NFCorpus
python -m asyncval.eval_elk \
  --model_name ERNIE \
	--query_file dataset/NFCorpus/scibert_linked_queries.test.tsv \
	--candidate_dir dataset/NFCorpus-knowledge-corpus/knowledge_corpus_constrainTuis.jsonl \
	--ckpts_dir output/model_nfc/ERNIE/tie_constrainTuis_from_coCondenser_24 \
	--tokenizer_name_or_path pretrained_model/co-condenser-marco \
	--qrel_file dataset/NFCorpus/qrels.test.tsv \
  --encoded_corpus_save_path precompute/NFCorpus/ERNIE \
  --untie_encoder False \
  --qry_use_ernie True \
	--metrics R@20 R@100 R@1000 nDCG@10 nDCG@20\
	--per_device_eval_batch_size 24 \
	--retrieve_batch_size -1 \
	--ent_hidden_size 100 \
  --depth 1000 \
  --fp16 \
  --q_max_len 12 \
  --p_max_len 510 \
  --report_to tensorboard \
  --dataloader_num_workers 10 \
  --logging_dir logs/NFCorpus/ERNIE/debug/test \
	--output_dir result/NFCorpus/ERNIE/debug/test





