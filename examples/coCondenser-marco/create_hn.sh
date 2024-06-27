python build_train_hn.py \
  --tokenizer_name Luyu/condenser \
  --depth 1000 \
  --n_sample 1000 \
  --hn_file ../../result/trec-covid/Condenser/ft1/Condenser.train.ckpt2400.txt \
  --qrels ../../dataset/trec-covid/qrels.train.tsv \
  --queries ../../dataset/trec-covid/train_queries.jsonl \
  --collection ../../dataset/trec-covid-corpus/corpus.jsonl \
  --save_to ../../dataset/trec-covid/train-hn_Condenser
