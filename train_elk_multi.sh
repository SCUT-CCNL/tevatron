export CUDA_VISIBLE_DEVICES=1

### --untie_encoder: True: 不共享参数, False: 共享（默认）
### qry_use_ernie: True: qry_use_encoder使用ERNIE, False: 不使用（默认）
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
# ELK_Multi: 查询侧：CLS表征 concat 平均实体表征，文档侧：CLS表征 分别concat （注意力实体表征、多个实体类型聚簇表征）
# ELK_Multi_V2：查询侧：CLS表征，文档侧：CLS表征、多个实体类型聚簇表征
# ELK_Multi_V3：将CLS、p_hidden和p_ent_hidden进行融合，然后再计算实体类型聚簇表征
# ELK_Multi: 类似V2，只不过实体聚簇表征通过注意力加权得到，不再是求均值 (不加ELK的那种全部实体的参与的注意力表征)




## TREC-COVID
## ELK_Multi + 其他多向量方法:
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
## ELK_Multi + 其他多向量方法:
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






















