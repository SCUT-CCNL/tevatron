export CUDA_VISIBLE_DEVICES=1

### --untie_encoder: True: 不共享参数, False: 共享（默认）
### qry_use_elk: True: qry_use_encoder使用知识编码器, False: 不使用（默认）
###   --dataloader_num_workers 6 \

## TREC-COVID
#python examples/elk/train_elk.py \
#  --output_dir output/model_covid/ERNIE_PubMedBERT \
#  --model_name_or_path bert-base-uncased123 \
#  --elk_model_path pretrained_model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
#  --tokenizer_name pretrained_model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
#  --ent_embedding_path kg_embedding/trec-covid/ent_embeddings_covid19_100d.pt \
#  --save_steps 900 \
#  --untie_encoder False \
#  --qry_use_elk True \
#  --dataset_name Tevatron/msmarco-passage-elk \
#  --train_dir  dataset/trec-covid/train_ERNIE_constrainTuis_qryent.jsonl \
#  --fp16 \
#  --ent_hidden_size 100 \
#  --per_device_train_batch_size 2 \
#  --train_n_passages 4 \
#  --learning_rate 1e-5 \
#  --q_max_len 12 \
#  --p_max_len 400 \
#  --num_train_epochs 1000 \
#  --logging_steps 50 \
#  --dataloader_num_workers 12 \
#  --gradient_accumulation_steps 2 \
#  --overwrite_output_dir \
#  --temperature 1.0 \
#  --comment from_pubmedbert_init \
#  --seed 42



## NFCorpus
#python examples/elk/train_elk.py \
#  --output_dir output/model_nfc/ERNIE_PubMedBERT \
#  --model_name_or_path bert-base-uncased123 \
#  --elk_model_path pretrained_model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
#  --tokenizer_name pretrained_model/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
#  --ent_embedding_path kg_embedding/nfcorpus/ent_embeddings_nfc_100d.pt \
#  --save_steps 2800 \
#  --untie_encoder False \
#  --qry_use_elk True \
#  --dataset_name Tevatron/msmarco-passage-elk \
#  --train_dir dataset/NFCorpus/train_ERNIE_constrainTuis_qryent.jsonl \
#  --ent_hidden_size 100 \
#  --per_device_train_batch_size 2 \
#  --train_n_passages 4 \
#  --learning_rate 1e-5 \
#  --fp16 \
#  --q_max_len 12 \
#  --p_max_len 400 \
#  --num_train_epochs 100 \
#  --logging_steps 300 \
#  --dataloader_num_workers 12 \
#  --gradient_accumulation_steps 2 \
#  --overwrite_output_dir \
#  --temperature 1.0 \
#  --comment elk_from_pubmedbert_init \
#  --seed 42
