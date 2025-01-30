import os
import argparse 
import json
from tqdm import tqdm
from reasoners.algorithm import MCTS
from reasoners.kbqa.agent import AgentWorldModel, AgentConfig, visualize_mcts_save, visualize_mcts_out
from reasoners import Reasoner
from reasoners.kbqa.pre_prompt import agent_prompt_kbqa
from reasoners.kbqa.prompt_function import prompt_function,execute_function_list
from reasoners.simcse import SimCSE
from utils.components.utils import dump_json
import copy
import random
import numpy as np
random.seed(1)
retrieval_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--llm_simulate_name", type=str, help="Name of the llm", default="8101/simulate")
parser.add_argument("--llm_reward_name", type=str, help="Name of the llm", default="8102/reward")
parser.add_argument("--base",type=str,help="Llama-3.1-8B-Instruct", default="test")
parser.add_argument("--task",type=str,help="task_name", default="test")
parser.add_argument("--dataset",type=str,help="task_name", default="WebQSP")
args = parser.parse_args()

para_configs = {
    "WebQSP":{
        "mcts_iters": 6,
        "deapth_limit": 15,
        "beam_size": 1,
        "entity_topk": 1,
        "entity_threshold": 0.0,
        "relation_topk": 3,
        "relation_threshold": 0.0,
        "limit_topk": 2,
        "limit_threshold": 0.0,
        "step_topk": 3,
        "step_threshold": -np.inf,
        "noway_panelty": -100.0,
        "explore_rate": 10,
        "reward_alpha": 0.5
    },
    "GrailQA":{
        "mcts_iters": 6,
        "deapth_limit": 20,
        "beam_size": 1,
        "entity_topk": 1,
        "entity_threshold": 0.0,
        "relation_topk": 3,
        "relation_threshold": 0.0,
        "limit_topk": 10,
        "limit_threshold": 0.0,
        "step_topk": 3,
        "step_threshold": -np.inf,       
        "noway_panelty": -100.0,
        "explore_rate": 10,
        "reward_alpha": 0.5
    },
    "GraphQ":{
        "mcts_iters": 6,
        "deapth_limit": 20,
        "beam_size": 2,
        "entity_topk": 1,
        "entity_threshold": 0.0,
        "relation_topk": 5,
        "relation_threshold": 0.0,
        "limit_topk": 10,
        "limit_threshold": 0.0,
        "step_topk": 5,
        "step_threshold": -np.inf,       
        "noway_panelty": -100.0,
        "explore_rate": 10,
        "reward_alpha": 0.5
    }
}

def run_kbqa():
    llm_simulate = f'http://localhost:{args.llm_simulate_name}'
    llm_reward = f'http://localhost:{args.llm_reward_name}'
    base_model = {'simulate':llm_simulate, 'reward':llm_reward}
    dataset = args.dataset
    if args.task=='explore':
        ex = True
    else:
        ex = False
    
    kbqa_data = json.load(open(f'dataset/{dataset}/processed/{dataset}_{"train" if ex else "test"}.json'))       
    save_kbqa_data = []
    os.makedirs(f'expr/KBQA/{args.base}/{dataset}/output', exist_ok=True)
    save_path = f'expr/KBQA/{args.base}/{dataset}/output/KBQA_{dataset}_{"explore" if ex else "test"}.json'
        
    prompt = para_configs[dataset].copy()
    prompt['ex'] = ex
    from reasoners.kbqa.limit import name_relation_list, tc_time_list, join_ban_relation_list, literal_relation_list, relation_list
    prompt['name_relation_list'] = name_relation_list
    prompt['tc_time_list'] = tc_time_list
    prompt['join_ban_relation_list'] = join_ban_relation_list
    prompt['literal_relation_list'] = literal_relation_list
    prompt['relation_list'] = relation_list
    
    if ex:
        prompt['beam_size'] = 2
        prompt['explore_rate'] = 50
        
    if not ex:
        kbqa_data = kbqa_data[:200]
    else:
        random.shuffle(kbqa_data)
    for row in tqdm(kbqa_data):
        
        prompt['kbqa_icl'] = agent_prompt_kbqa.format(examples = '', question = row["question"], scratchpad = '')
        row['function_prompt'], row['entities'] = prompt_function(row['function_list'])
        row['dataset'] = dataset   
        
        print(row['question'])
        world_model = AgentWorldModel(base_model=base_model, retrieval_model=retrieval_model, prompt=prompt, max_steps=prompt['deapth_limit'])
        config = AgentConfig(base_model=base_model, retrieval_model=retrieval_model, prompt=prompt, reward_alpha=prompt['reward_alpha'])
        algorithm = MCTS(depth_limit=prompt['deapth_limit'], disable_tqdm=False, output_trace_in_each_iter=True, n_iters=prompt['mcts_iters'], w_exp = prompt['explore_rate'], cum_reward=np.mean, calc_q=max) # 
        reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)
        result_rap = reasoner_rap(row)
        
        row['pred_function_prompt'] = result_rap.terminal_state.blocks_state
        row['pred_function_list'] = result_rap.terminal_state.functions_state
        result = execute_function_list(result_rap.terminal_state.functions_state,dataset)
        row['pred_sexpr'] = result['sexpr']
        row['pred_answer'] = result['answers']
        print('\n'.join(row['function_list'])) 
        print('\n'.join(result_rap.terminal_state.functions_state))
        print(row['sexpr'])
        print(row['pred_sexpr'])
        print()
        print(row['answer'])
        print(row['pred_answer'])                
                
        save_kbqa_data.append(copy.deepcopy(row))
        dump_json(save_kbqa_data,save_path,indent=4)
        row['result_log'] = visualize_mcts_save(result_rap)
        # visualize_mcts_out(row['result_log'])
        
        if not ex:
            if row['dataset'] == 'WebQSP':
                from utils.evaluation.webqsp_evaluate import webqsp_evaluate_valid_results
                webqsp_evaluate_valid_results(save_path,save_path)
            elif row['dataset'] == 'GrailQA':
                from utils.evaluation.grailqa_evaluate import grailqa_evaluate_valid_results
                grailqa_evaluate_valid_results(save_path,save_path) 
            elif row['dataset'] == 'GraphQ':
                from utils.evaluation.webqsp_evaluate import graphq_evaluate_valid_results
                graphq_evaluate_valid_results(save_path,save_path)          
                
      
        
if __name__ == '__main__':
    run_kbqa()