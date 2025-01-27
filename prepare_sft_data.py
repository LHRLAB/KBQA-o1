import argparse 
import json
from tqdm import tqdm
import random
import copy
from reasoners.kbqa.prompt_function import prompt_function
from reasoners.kbqa.pre_prompt import agent_prompt_kbqa
from src.upload_data import sft_data_to_json, upload_sft_data
random.seed(42)

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--dataset",type=str,default="WebQSP",help="task name")
args = parser.parse_args()

def prepare_sft_KBQA(dataset):
    sft_KBQA_simulate_data = []
    sft_KBQA_reward_data = []
    assert dataset in ['WebQSP','GrailQA', 'GraphQ']
    print(f'Processing {dataset} dataset...')
    kbqa_data = json.load(open(f'dataset/{dataset}/processed/{dataset}_train.json')) 
    random.shuffle(kbqa_data)
    sample_dict = {"WebQSP": 100, "GrailQA":40, "GraphQ": 100}
    for row in tqdm(kbqa_data[:sample_dict[dataset]]):
        row['dataset'] = copy.deepcopy(dataset)
        row['function_prompt'], row['entities'] = prompt_function(row['function_list'])

        input = agent_prompt_kbqa.format(examples = '', question = row["question"], scratchpad = '')
        if 'Thought' in row['question'] or 'thought' in row['question']:
            continue
        blocks = ['Thought'+block for block in row['function_prompt'].split('Thought')[1:]]
        for cnt, block in enumerate(blocks):
            simulate_input = input + ''.join(blocks[:cnt])
            simulate_output = ''.join(blocks[cnt:])
            sft_KBQA_simulate_data.append(sft_data_to_json(simulate_input,simulate_output))
        reward_input = row['question']
        reward_output = row['sexpr']
        for ent,ent_id in row['entities']:
            if ent_id in reward_output and ent:
                reward_output = reward_output.replace(f' {ent_id} ',f' <{ent.replace(" ","_")}> ')
                reward_output = reward_output.replace(f' {ent_id})',f' <{ent.replace(" ","_")}>)')
        sft_KBQA_reward_data.append(sft_data_to_json(reward_input,reward_output))
        
    print("Simulate data length: ", len(sft_KBQA_simulate_data))
    print("Reward data length: ", len(sft_KBQA_reward_data))
    
    upload_sft_data(sft_KBQA_simulate_data, f'sft_KBQA_{dataset}_simulate')
    upload_sft_data(sft_KBQA_reward_data, f'sft_KBQA_{dataset}_reward')
     
if __name__ == '__main__':
    prepare_sft_KBQA(args.dataset)