import json
import os

def sft_data_to_json(input,output):
    sft_data_json = {
        "conversation": [
            {
                "from": "human",
                "value": input,
            },
            {   
                "from": "gpt",
                "value": output
            }
        ]
    }
    return sft_data_json

def upload_sft_data(sft_data, task_name):
    os.makedirs(f'data', exist_ok=True)
    with open(f'data/{task_name}.json', encoding='utf-8', mode='w') as f:
        json.dump(sft_data, f, indent=2)
    try:         
        if os.path.exists(f'data/dataset_info.json'):
            with open('data/dataset_info.json', 'r') as file:
                dataset_info = json.load(file) 
        else:
            dataset_info = dict()
        dataset_info[task_name] = {
            "file_name": f'{task_name}.json',
            "formatting": "sharegpt",
            "columns": {
            "messages": "conversation",
            }
        }   
        with open('data/dataset_info.json', encoding='utf-8', mode='w') as f:
            json.dump(dataset_info, f, indent=2)  
    except:
        print("Dead Lock Error: dataset_info.json")
    print(f'sft_dir: data/{task_name}.json')
    print(f"data size: {len(sft_data)}")
    
    
def dpo_data_to_json(input,chosen,rejected):
    dpo_data_json = {
        "conversation": [
            {
                "from": "human",
                "value": input,
            }
        ],
        "chosen": 
            {   
                "from": "gpt",
                "value": chosen
            },
        "rejected": 
            {   
                "from": "gpt",
                "value": rejected
            }
    }
    return dpo_data_json

def upload_dpo_data(dpo_data, task_name):
    os.makedirs(f'data', exist_ok=True)
    with open(f'data/{task_name}.json', encoding='utf-8', mode='w') as f:
        json.dump(dpo_data, f, indent=2) 
    try:
        if os.path.exists(f'data/dataset_info.json'):
            with open('data/dataset_info.json', 'r') as file:
                dataset_info = json.load(file) 
        else:
            dataset_info = dict()
        dataset_info[task_name] = {
            "file_name": f'{task_name}.json',
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
            "messages": "conversation",
            "chosen": "chosen",
            "rejected": "rejected"
            }
        }   
        with open('data/dataset_info.json', encoding='utf-8', mode='w') as f:
            json.dump(dataset_info, f, indent=2) 
    except:
        print("Dead Lock Error: dataset_info.json")
    print(f'dpo_dir: data/{task_name}.json')
    print(f"data size: {len(dpo_data)}") 