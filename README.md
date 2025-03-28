# KBQA-o1

KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search [[paper](https://arxiv.org/abs/2501.18922)]

## Overview

![](./figs/case.png)

## Preparation

#### Step1: Conda Environment
```bash
conda create -n kbqao1 python=3.11
conda activate kbqao1
pip install torch==2.3.0
pip install -r requirements.txt
sudo apt install unixodbc
export PYTHONPATH=$PWD
```

#### Step2: Setup Freebase KG Service

Below steps are according to [Freebase Virtuoso Setup](https://github.com/dki-lab/Freebase-Setup). 

(1) Clone from `dki-lab/Freebase-Setup`:
```bash
cd Freebase-Setup
```
(2) Processed [Freebase](https://developers.google.com/freebase) Virtuoso DB file can be downloaded from [here](https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip) or via wget (WARNING: 53G+ disk space is needed):
```bash
tar -zxvf virtuoso_db.zip
```
(3) Managing the Virtuoso service:
To start service:
```bash
chmod +x virtuoso-opensource/bin/virtuoso-t
python3 virtuoso.py start 3001 -d virtuoso_db
```
and to stop a currently running service at the same port:
```bash
chmod +x virtuoso-opensource/bin/isql
python3 virtuoso.py stop 3001
```
A server with at least 100 GB RAM is recommended.

#### Step3: Download Freebase Ontology Files
- Download `fb_roles`, `fb_types`, `reverse_properties` from [here](https://github.com/dki-lab/GrailQA/tree/main/ontology) to `dataset/Freebase/`.
```
KBQA-o1/
└── dataset/
    ├── Freebase/   
        ├── fb_roles
        ├── fb_types
        └── reverse_properties                                                            
```

#### Step4: Download KBQA Datasets

Experiments are conducted on 3 classical KBQA benchmarks: WebQSP, GrailQA and GraphQ.
- **WebQSP**: Download the WebQSP dataset from [here](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/) and put them under `dataset/WebQSP/origin`. The dataset files should be named as `WebQSP.test[train].json`.
- **GrailQA**: Download the GrailQA dataset [here](https://dki-lab.github.io/GrailQA/) and put them under both `dataset/GrailQA/origin`. The dataset files should be named as `grailqa_v1.0_test_public[train,dev].json`.
- **GraphQ**: Download the GraphQ dataset [here](https://github.com/dki-lab/GrailQA/tree/main/data) and put them under both `dataset/GraphQ/origin`. The dataset files should be named as `graphquestions_v1_fb15_test[training]_091420.json`.

```
KBQA-o1/
└── dataset/
    ├── WebQSP/                  
        ├── origin/                    
            ├── WebQSP.train.json                    
            └── WebQSP.test.json 
    ├── GrailQA/                 
        ├── origin/                    
            ├── grailqa_v1.0_train.json                   
            ├── grailqa_v1.0_dev.json      
            └── grailqa_v1.0_test_public.json  
    ├── GraphQ/                 
        ├── origin/                    
            ├── graphquestions_v1_fb15_training_091420.json
            └── graphquestions_v1_fb15_test_091420.json                                       
```

#### Step5: Preprocess KBQA Datasets

Parse SPARQL queries to S-expressions and Function-lists.
- **WebQSP**: Run `python data_process.py --dataset WebQSP` and the merged data file will be saved as `dataset/WebQSP/processed/WebQSP_train[test].json`.
- **GrailQA**: Run `python data_process.py --dataset GrailQA` and the merged data file will be saved as `dataset/GrailQA/processed/GrailQA_train[test,test_public].json`.
- **GraphQ**: Run `python data_process.py --dataset GraphQ` and the merged data file will be saved as `dataset/GraphQ/processed/GraphQ_train[test].json`.

```
KBQA-o1/
└── dataset/
    ├── WebQSP/                  
        ├── processed/                    
            ├── WebQSP_train.json                    
            └── WebQSP_test.json  
    ├── GrailQA/                 
        ├── processed/                    
            ├── GrailQA_train.json   
            └── GrailQA_test.json                 
    ├── GraphQ/                 
        ├── processed/                    
            ├── GraphQ_train.json                    
            └── GraphQ_test.json                                        
```

## KBQA-o1

### WebQSP

#### Step1: Prepare SFT Data for KBQA

```bash
python prepare_sft_data.py --dataset WebQSP
```

#### Step2: SFT Training Simulate Model and Reward Model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --dataset sft_KBQA_WebQSP_simulate --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_simulate/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 50.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft_KBQA_WebQSP_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --use_dora --dataset sft_KBQA_WebQSP_reward --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_reward/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 100.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft_KBQA_WebQSP_reward.log 2>&1 &
```

#### Step3: Export 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_simulate/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_simulate/export_model/simulate --export_size 2 --export_legacy_format False
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_reward/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_reward/export_model/reward --export_size 2 --export_legacy_format False
```

#### Step4: Start LLM API for 2 Models
```bash
CUDA_VISIBLE_DEVICES=0 API_PORT=8101 MODEL_NAME=simulate nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_simulate/export_model/simulate --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft_KBQA_WebQSP_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 API_PORT=8102 MODEL_NAME=reward nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_reward/export_model/reward --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft_KBQA_WebQSP_reward.log 2>&1 &
```

#### Step5: Explore KBQA
```bash
CUDA_VISIBLE_DEVICES=0 nohup python run_explore.py --llm_simulate_name 8101/simulate --llm_reward_name 8102/reward --base Llama-3.1-8B-Instruct --task explore --dataset WebQSP >> result_Llama-3.1-8B-Instruct_explore_KBQA_WebQSP_sft.log 2>&1 &
```

#### Step6: Prepare SFT2 Data for KBQA
```bash
python prepare_sft2_data.py --llm_reward_name 8102/reward --base Llama-3.1-8B-Instruct --dataset WebQSP --limit "30"
``` 

#### Step7: End LLM API for 2 Models
```bash
bash utils/kill_llm_api_WebQSP.sh
```

#### Step8: SFT2 Training 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_simulate/export_model/simulate --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --dataset sft2_KBQA_WebQSP_simulate --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_simulate/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 10.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft2_KBQA_WebQSP_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_reward/export_model/reward --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --use_dora --dataset sft2_KBQA_WebQSP_reward --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_reward/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 20.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft2_KBQA_WebQSP_reward.log 2>&1 &
```

#### Step9: Export 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_simulate/export_model/simulate --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_simulate/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_simulate/export_model/simulate --export_size 2 --export_legacy_format False
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft_KBQA_WebQSP_reward/export_model/reward --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_reward/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_reward/export_model/reward --export_size 2 --export_legacy_format False
```

#### Step10: Start LLM API for 2 Models
```bash
CUDA_VISIBLE_DEVICES=0 API_PORT=8101 MODEL_NAME=simulate nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_simulate/export_model/simulate --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft2_KBQA_WebQSP_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 API_PORT=8102 MODEL_NAME=reward nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/WebQSP/sft2_KBQA_WebQSP_reward/export_model/reward --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft2_KBQA_WebQSP_reward.log 2>&1 &
```

#### Step11: Test KBQA
```bash
CUDA_VISIBLE_DEVICES=0 nohup python run_explore.py --llm_simulate_name 8101/simulate --llm_reward_name 8102/reward --base Llama-3.1-8B-Instruct --task test --dataset WebQSP >> result_Llama-3.1-8B-Instruct_test_KBQA_WebQSP_sft2.log 2>&1 &
```

#### Step12: End LLM API for 2 Models
```bash
bash utils/kill_llm_api_WebQSP.sh
```

### GrailQA

#### Step1: Prepare SFT Data for KBQA
```bash
python prepare_sft_data.py --dataset GrailQA
```

#### Step2: SFT Training Simulate Model and Reward Model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --dataset sft_KBQA_GrailQA_simulate --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_simulate/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 100.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft_KBQA_GrailQA_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --use_dora --dataset sft_KBQA_GrailQA_reward --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_reward/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 300.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft_KBQA_GrailQA_reward.log 2>&1 &
```

#### Step3: Export 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_simulate/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_simulate/export_model/simulate --export_size 2 --export_legacy_format False
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_reward/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_reward/export_model/reward --export_size 2 --export_legacy_format False
```

#### Step4: Start LLM API for 2 Models
```bash
CUDA_VISIBLE_DEVICES=1 API_PORT=8103 MODEL_NAME=simulate nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_simulate/export_model/simulate --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft_KBQA_GrailQA_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 API_PORT=8104 MODEL_NAME=reward nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_reward/export_model/reward --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft_KBQA_GrailQA_reward.log 2>&1 &
```

#### Step5: Explore KBQA
```bash
CUDA_VISIBLE_DEVICES=1 nohup python run_explore.py --llm_simulate_name 8103/simulate --llm_reward_name 8104/reward --base Llama-3.1-8B-Instruct --task explore --dataset GrailQA >> result_Llama-3.1-8B-Instruct_explore_KBQA_GrailQA_sft.log 2>&1 &
```

#### Step6: Prepare SFT2 Data for KBQA
```bash
python prepare_sft2_data.py --llm_reward_name 8104/reward --base Llama-3.1-8B-Instruct --dataset GrailQA --limit "-100"
``` 

#### Step7: End LLM API for 2 Models
```bash
bash utils/kill_llm_api_GrailQA.sh
```

#### Step8: SFT2 Training 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_simulate/export_model/simulate --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --dataset sft2_KBQA_GrailQA_simulate --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_simulate/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 10.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft2_KBQA_GrailQA_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_reward/export_model/reward --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --use_dora --dataset sft2_KBQA_GrailQA_reward --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_reward/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 20.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft2_KBQA_GrailQA_reward.log 2>&1 &
```

#### Step9: Export 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_simulate/export_model/simulate --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_simulate/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_simulate/export_model/simulate --export_size 2 --export_legacy_format False
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft_KBQA_GrailQA_reward/export_model/reward --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_reward/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_reward/export_model/reward --export_size 2 --export_legacy_format False
```

#### Step10: Start LLM API for 2 Models
```bash
CUDA_VISIBLE_DEVICES=1 API_PORT=8103 MODEL_NAME=simulate nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_simulate/export_model/simulate --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft2_KBQA_GrailQA_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 API_PORT=8104 MODEL_NAME=reward nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GrailQA/sft2_KBQA_GrailQA_reward/export_model/reward --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft2_KBQA_GrailQA_reward.log 2>&1 &
```

#### Step11: Test KBQA
```bash
CUDA_VISIBLE_DEVICES=1 nohup python run_explore.py --llm_simulate_name 8103/simulate --llm_reward_name 8104/reward --base Llama-3.1-8B-Instruct --task test --dataset GrailQA >> result_Llama-3.1-8B-Instruct_test_KBQA_GrailQA_sft2.log 2>&1 &
```

#### Step12: End LLM API for 2 Models
```bash
bash utils/kill_llm_api_GrailQA.sh
```




### GraphQ

#### Step1: Prepare SFT Data for KBQA

```bash
python prepare_sft_data.py --dataset GraphQ
```

#### Step2: SFT Training Simulate Model and Reward Model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --dataset sft_KBQA_GraphQ_simulate --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_simulate/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 50.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft_KBQA_GraphQ_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --use_dora --dataset sft_KBQA_GraphQ_reward --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_reward/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 100.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft_KBQA_GraphQ_reward.log 2>&1 &
```

#### Step3: Export 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_simulate/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_simulate/export_model/simulate --export_size 2 --export_legacy_format False
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_reward/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_reward/export_model/reward --export_size 2 --export_legacy_format False
```

#### Step4: Start LLM API for 2 Models
```bash
CUDA_VISIBLE_DEVICES=2 API_PORT=8105 MODEL_NAME=simulate nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_simulate/export_model/simulate --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft_KBQA_GraphQ_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 API_PORT=8106 MODEL_NAME=reward nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_reward/export_model/reward --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft_KBQA_GraphQ_reward.log 2>&1 &
```

#### Step5: Explore KBQA
```bash
CUDA_VISIBLE_DEVICES=2 nohup python run_explore.py --llm_simulate_name 8105/simulate --llm_reward_name 8106/reward --base Llama-3.1-8B-Instruct --task explore --dataset GraphQ >> result_Llama-3.1-8B-Instruct_explore_KBQA_GraphQ_sft.log 2>&1 &
```

#### Step6: Prepare SFT2 Data for KBQA
```bash
python prepare_sft2_data.py --llm_reward_name 8106/reward --base Llama-3.1-8B-Instruct --dataset GraphQ --limit "-50"
``` 

#### Step7: End LLM API for 2 Models
```bash
bash utils/kill_llm_api_GraphQ.sh
```

#### Step8: SFT2 Training 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_simulate/export_model/simulate --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --dataset sft2_KBQA_GraphQ_simulate --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_simulate/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 10.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft2_KBQA_GraphQ_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus 4 --master_port=9902 src/train.py --deepspeed ds_config.json --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_reward/export_model/reward --stage sft --do_train --finetuning_type lora --lora_target q_proj,v_proj --use_dora --dataset sft2_KBQA_GraphQ_reward --template llama3 --cutoff_len 1024 --overwrite_cache --preprocessing_num_workers 16 --output_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_reward/checkpoint  --logging_steps 10 --save_steps 10000 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_train_epochs 20.0 --lr_scheduler_type cosine --bf16 >> result_Llama-3.1-8B-Instruct_sft2_KBQA_GraphQ_reward.log 2>&1 &
```

#### Step9: Export 2 Models
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_simulate/export_model/simulate --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_simulate/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_simulate/export_model/simulate --export_size 2 --export_legacy_format False
```
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/export_model.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft_KBQA_GraphQ_reward/export_model/reward --adapter_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_reward/checkpoint --template llama3 --finetuning_type lora --use_dora --export_dir expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_reward/export_model/reward --export_size 2 --export_legacy_format False
```

#### Step10: Start LLM API for 2 Models
```bash
CUDA_VISIBLE_DEVICES=2 API_PORT=8105 MODEL_NAME=simulate nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_simulate/export_model/simulate --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft2_KBQA_GraphQ_simulate.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 API_PORT=8106 MODEL_NAME=reward nohup python src/llm_api.py --model_name_or_path expr/KBQA/Llama-3.1-8B-Instruct/GraphQ/sft2_KBQA_GraphQ_reward/export_model/reward --template llama3 --temperature 0.0 >> result_Llama-3.1-8B-Instruct_llm_api_sft2_KBQA_GraphQ_reward.log 2>&1 &
```

#### Step11: Test KBQA
```bash
CUDA_VISIBLE_DEVICES=2 nohup python run_explore.py --llm_simulate_name 8105/simulate --llm_reward_name 8106/reward --base Llama-3.1-8B-Instruct --task test --dataset GraphQ >> result_Llama-3.1-8B-Instruct_test_KBQA_GraphQ_sft2.log 2>&1 &
```

#### Step12: End LLM API for 2 Models
```bash
bash utils/kill_llm_api_GraphQ.sh
```

## BibTex

If you find this work is helpful for your research, please cite:

```bibtex
@misc{luo2025kbqao1,
      title={KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search}, 
      author={Haoran Luo and Haihong E and Yikai Guo and Qika Lin and Xiaobao Wu and Xinyu Mu and Wenhao Liu and Meina Song and Yifan Zhu and Luu Anh Tuan},
      year={2025},
      eprint={2501.18922},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.18922}, 
}
```

For further questions, please contact: luohaoran@bupt.edu.cn.

## Acknowledgement

This repo benefits from [KB-Coder](https://github.com/Arthurizijar/KB-Coder), [LLM-Reasoners](https://github.com/maitrix-org/llm-reasoners) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their wonderful works.