export CUDA_VISIBLE_DEVICES=1

### --untie_encoder: True: ���������, False: ����Ĭ�ϣ�
### qry_use_ernie: True: qry_use_encoderʹ��ERNIE, False: ��ʹ�ã�Ĭ�ϣ�
###   --dataloader_num_workers 6 \

## TREC-COVID
#nohup python examples/ernie_multi/train_ernie_multi.py \
#  --output_dir output/model_covid/ELK_Multi_V4/v10/max5 \
#  --model_name_or_path bert-base-uncased123 \
#  --ernie_model_path pretrained_model/co-condenser-marco \
#  --tokenizer_name pretrained_model/co-condenser-marco \
#  --ent_embedding_path kg_embedding/trec-covid/ent_embeddings_covid19_100d.pt \
#  --save_steps 900 \
#  --untie_encoder False \
#  --qry_use_ernie True \
#  --dataset_name Tevatron/msmarco-passage-ernie-multi \
#  --train_dir  dataset/trec-covid/train_ERNIE_constrainTuis_qryent_with_types.jsonl \
#  --ent_cluster_dir dataset/trec-covid-cluster/trec_covid_doc_with_ent_cluster_max5.jsonl \
#  --fp16 \
#  --ent_hidden_size 100 \
#  --per_device_train_batch_size 2 \
#  --train_n_passages 4 \
#  --learning_rate 1e-5 \
#  --q_max_len 12 \
#  --p_max_len 400 \
#  --num_train_epochs 1000 \
#  --logging_steps 300 \
#  --dataloader_num_workers 10 \
#  --gradient_accumulation_steps 2 \
#  --overwrite_output_dir \
#  --temperature 1.0 \
#  --comment ELK_Multi_V4_Save5RepsForEachDocument_ProjectToDim128  \
#  --seed 42 \
#  > nohup.log &



## NFCorpus
nohup python examples/ernie_multi/train_ernie_multi.py \
  --output_dir output/model_nfc/ELK_Multi_V4/v6/max5 \
  --model_name_or_path bert-base-uncased123 \
  --ernie_model_path pretrained_model/co-condenser-marco \
  --tokenizer_name pretrained_model/co-condenser-marco \
  --ent_embedding_path kg_embedding/nfcorpus/ent_embeddings_nfc_100d.pt \
  --save_steps 2800 \
  --untie_encoder False \
  --qry_use_ernie True \
  --dataset_name Tevatron/msmarco-passage-ernie-multi \
  --train_dir dataset/NFCorpus/train_ERNIE_constrainTuis_qryent_with_types.jsonl \
  --ent_cluster_dir dataset/NFCorpus-cluster/nfcorpus_doc_with_ent_cluster_max5.jsonl \
  --fp16 \
  --ent_hidden_size 100 \
  --per_device_train_batch_size 2 \
  --train_n_passages 4 \
  --learning_rate 1e-5 \
  --q_max_len 12 \
  --p_max_len 400 \
  --num_train_epochs 200 \
  --logging_steps 300 \
  --dataloader_num_workers 10 \
  --gradient_accumulation_steps 2 \
  --overwrite_output_dir \
  --temperature 1.0 \
  --comment ELK_Multi_V4_Save5RepsForEachDocument_ProjectToDim128 \
  --seed 42 \
  > nohup_v6.log &


# addPsgAttnPoolingEntRep
# ELK_Multi: ��ѯ�ࣺCLS���� concat ƽ��ʵ��������ĵ��ࣺCLS���� �ֱ�concat ��ע����ʵ����������ʵ�����;۴ر�����
# ELK_Multi_V2����ѯ�ࣺCLS�������ĵ��ࣺCLS���������ʵ�����;۴ر���
# ELK_Multi_V3����CLS��p_hidden��p_ent_hidden�����ںϣ�Ȼ���ټ���ʵ�����;۴ر���
# ELK_Multi: ����V2��ֻ����ʵ��۴ر���ͨ��ע������Ȩ�õ������������ֵ (����ELK������ȫ��ʵ��Ĳ����ע��������)




## TREC-COVID
## ELK_Multi + ��������������:
#python examples/elk_multi_variants/elk_multi_eva_multi/train_elk_multi_eva_multi.py \
#  --output_dir output/model_covid/ELK_Multi_EVA_Multi \
#  --model_name_or_path bert-base-uncased123 \
#  --ernie_model_path pretrained_model/co-condenser-marco \
#  --tokenizer_name pretrained_model/co-condenser-marco \
#  --ent_embedding_path kg_embedding/trec-covid/eva_cluster_embedding_covid19_100d.pt \
#  --save_steps 900 \
#  --untie_encoder False \
#  --qry_use_ernie True \
#  --dataset_name Tevatron/msmarco-passage-ernie-multi-eva-multi \
#  --train_dir  dataset/trec-covid/train_ERNIE_constrainTuis_qryent_with_types.jsonl \
#  --eva_cluster_dir dataset/trec-covid-cluster/trec_covid_eva_cluster_for_doc.jsonl \
#  --fp16 \
#  --ent_hidden_size 100 \
#  --per_device_train_batch_size 2 \
#  --train_n_passages 4 \
#  --learning_rate 1e-5 \
#  --q_max_len 12 \
#  --p_max_len 400 \
#  --num_train_epochs 1000 \
#  --logging_steps 300 \
#  --dataloader_num_workers 10 \
#  --gradient_accumulation_steps 2 \
#  --overwrite_output_dir \
#  --temperature 1.0 \
#  --comment ELK_Multi_with_EVA_Multi \
#  --seed 42

## NFCorpus
## ELK_Multi + ��������������:
#python examples/elk_multi_variants/elk_multi_eva_multi/train_elk_multi_eva_multi.py \
#  --output_dir output/model_nfc/ELK_Multi_EVA_Multi/\
#  --model_name_or_path bert-base-uncased123 \
#  --ernie_model_path pretrained_model/co-condenser-marco \
#  --tokenizer_name pretrained_model/co-condenser-marco \
#  --ent_embedding_path kg_embedding/nfcorpus/eva_cluster_embedding_nfc_100d.pt \
#  --save_steps 2800 \
#  --untie_encoder False \
#  --qry_use_ernie True \
#  --dataset_name Tevatron/msmarco-passage-ernie-multi-eva-multi \
#  --train_dir dataset/NFCorpus/train_ERNIE_constrainTuis_qryent_with_types.jsonl \
#  --eva_cluster_dir dataset/NFCorpus-cluster/nfcorpus_eva_cluster_for_doc.jsonl \
#  --fp16 \
#  --ent_hidden_size 100 \
#  --per_device_train_batch_size 2 \
#  --train_n_passages 4 \
#  --learning_rate 1e-5 \
#  --q_max_len 12 \
#  --p_max_len 400 \
#  --num_train_epochs 100 \
#  --logging_steps 300 \
#  --dataloader_num_workers 10 \
#  --gradient_accumulation_steps 2 \
#  --overwrite_output_dir \
#  --temperature 1.0 \
#  --comment ELK_Multi_with_EVA_Multi \
#  --seed 42






















