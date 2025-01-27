import argparse 
import json
from tqdm import tqdm
import requests
from src.upload_data import sft_data_to_json, upload_sft_data
from reasoners.kbqa.prompt_function import get_sexpr
from reasoners.kbqa.pre_prompt import agent_prompt_kbqa

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--llm_reward_name",type=str,default="8102/reward",help="base")
parser.add_argument("--base",type=str,default="Llama-3.1-8B-Instruct",help="base")
parser.add_argument("--dataset",type=str,default="WebQSP",help="dataset")
parser.add_argument("--limit",type=int,default=30,help="dataset")
parser.add_argument("--merge_simulate_dir",type=str,default="data/sft_KBQA_WebQSP_simulate.json",help="")
parser.add_argument("--merge_reward_dir",type=str,default="data/sft_KBQA_WebQSP_reward.json",help="")
args = parser.parse_args()

     
if __name__ == '__main__':
    base_model = f'http://localhost:{args.llm_reward_name}'
    sft2_KBQA_simulate_data = json.load(open(args.merge_simulate_dir)) 
    sft2_KBQA_reward_data = json.load(open(args.merge_reward_dir))
    print("Simulate data length: ", len(sft2_KBQA_simulate_data))
    print("Reward data length: ", len(sft2_KBQA_reward_data))
    assert args.dataset in ['WebQSP','GrailQA', 'GraphQ']
    print(f'Processing {args.dataset} dataset...')
    kbqa_data = json.load(open(f'expr/KBQA/{args.base}/{args.dataset}/output/KBQA_{args.dataset}_explore.json')) 
    for row in tqdm(kbqa_data):
        sexpr = get_sexpr(row['pred_function_prompt'],row['entities'])
        score = requests.post(base_model, json={ "input": row['question'], "output": [sexpr]}).json()[0]
        if score > args.limit and len(row['pred_answer']) > 0:
            input = agent_prompt_kbqa.format(examples = '', question = row["question"], scratchpad = '')
            if 'Thought' in row['question'] or 'thought' in row['question']:
                continue
            blocks = ['Thought'+block for block in row['pred_function_prompt'].split('Thought')[1:]]
            for cnt, block in enumerate(blocks):
                simulate_input = input + ''.join(blocks[:cnt])
                simulate_output = ''.join(blocks[cnt:])
                sft2_KBQA_simulate_data.append(sft_data_to_json(simulate_input,simulate_output))    
            sft2_KBQA_reward_data.append(sft_data_to_json(row['question'],sexpr))        
            if set(row['pred_answer']) != set(row['answer']):
                print(score)
                pass
        else:
            if set(row['pred_answer']) == set(row['answer']):
                pass            
        
    print("Simulate data length: ", len(sft2_KBQA_simulate_data))
    print("Reward data length: ", len(sft2_KBQA_reward_data))
    
    upload_sft_data(sft2_KBQA_simulate_data, f'sft2_KBQA_{args.dataset}_simulate')
    upload_sft_data(sft2_KBQA_reward_data, f'sft2_KBQA_{args.dataset}_reward')