from reasoners import WorldModel, LanguageModel, SearchConfig
from typing import NamedTuple
import copy
import requests
from reasoners.kbqa.pre_prompt import PLAN_LIST
from reasoners.kbqa.prompt_function import get_next_relations,get_next_r_relations,test_execute_function_list,ent_type,get_sexpr
import re
from tqdm import tqdm
AgentAction = str

class AgentState(NamedTuple):
    step_idx: int
    last_blocks_state: str
    blocks_state: str
    buffered_action: AgentAction
    expression_id: str
    functions_state: list


class AgentWorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 retrieval_model,
                 prompt: dict,
                 max_steps: int = 4,
                 batch_size: int = 1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.retrieval_model = retrieval_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> AgentState:
        return AgentState(step_idx=0, 
                          last_blocks_state="", 
                          blocks_state="", 
                          buffered_action="",
                          expression_id="",
                          functions_state=[])

    def step(self, state: AgentState, action: AgentAction) -> tuple[AgentState, dict]:
        step_idx = state.step_idx
        blocks_state = state.blocks_state + action[0]
        functions_state = state.functions_state + [action[1]]
        new_buffered_action = action[0]

        state = AgentState(step_idx=step_idx + 1,
                        last_blocks_state=state.blocks_state,
                        blocks_state=blocks_state,
                        buffered_action=new_buffered_action,
                        expression_id=action[1].split(" = ")[0].replace('expression',''),
                        functions_state=functions_state)
        return state

    def update_blocks(self, block_states: str, action: AgentAction) -> str:
        # if "pick" in action:
        #     key = "world_update_pickup"
        # elif "unstack" in action:
        #     key = "world_update_unstack"
        # elif "put" in action:
        #     key = "world_update_putdown"
        # elif "stack" in action:
        #     key = "world_update_stack"
        # else:
        #     raise ValueError("Invalid action")
        # world_update_prompt = self.prompt[key].format(block_states, action.capitalize() + ".")
        # world_output = self.base_model.generate([world_update_prompt],
        #                                         eos_token_id="\n",
        #                                         hide_input=True,
        #                                         temperature=0).text[0].strip()
        new_state = block_states + action
        # new_state = utils.apply_change(world_output, block_states)
        return new_state

    def is_terminal(self, state: AgentState) -> bool:
        if ': Finish ' in state.blocks_state and 'STOP(' in state.blocks_state:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False
    
    
    
class AgentConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 retrieval_model,
                 prompt: dict,
                 batch_size: int = 1,
                 reward_alpha: float = 0.5,
                 goal_reward_default: float = 0.,
                 goal_reached_reward: float = 100.) -> None:
        super().__init__()
        self.base_model = base_model
        self.retrieval_model = retrieval_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        
    def get_candidate_entities(self, state: AgentState):
        candidate_entities = copy.deepcopy(self.example['entities'])
        candidate_tc_time = []
        for ent,ent_id in candidate_entities:
            if ent_type(ent_id) == "int":
                if ent in self.prompt['tc_time_list']:
                    candidate_entities.remove((ent,ent_id))
                    candidate_tc_time.append((ent,ent_id))
        for ent_id in re.findall(f"START\(\'(.*?)\'\)", '\n'.join(state.functions_state)):
            for ent in candidate_entities:
                if ent_id == ent[1]:
                    candidate_entities.remove(ent)
                    break
        return candidate_entities
        
    def get_candidate_relations(self, state: AgentState):
        if 'START' in state.functions_state[-1]:
            ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
            if ent_type(ent) in ['name', 'literal', 'int', 'url', 'onto']:
                candidate_r_relations = []
                candidate_relations = []
                for rel in self.prompt['name_relation_list']:
                    if '(R ' in rel:
                        candidate_r_relations.append(re.findall(f"\(R (.*?)\)", rel)[0])
                    else:
                        candidate_relations.append(rel)
            else:
                candidate_r_relations = get_next_r_relations(state.functions_state,self.example['dataset'])
                candidate_relations = get_next_relations(state.functions_state,self.example['dataset'])
        else:
            candidate_r_relations = get_next_r_relations(state.functions_state,self.example['dataset'])
            candidate_relations = get_next_relations(state.functions_state,self.example['dataset'])
        
        candidate_relations = [(rel,f'(R {rel})') for rel in candidate_r_relations if rel not in self.prompt['join_ban_relation_list'] and rel in self.prompt['relation_list']] + [(rel,f'{rel}') for rel in candidate_relations if rel not in self.prompt['join_ban_relation_list'] and rel in self.prompt['relation_list']]             
        
        if self.example['dataset'] == 'WebQSP':
            if 'START' in state.functions_state[-1]:
                ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
                if ent_type(ent) != 'entity':   
                    candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if rel!='type.object.name']
            else:
                candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if rel!='type.object.name']
        return candidate_relations  
        

    def get_actions(self, state: AgentState):
        scratchpads_list = []
        step_n = state.step_idx + 1
        plans_dict = {plan: PLAN_LIST[plan].format(step_n,step_n) for plan in PLAN_LIST.keys()}

        inputs = self.prompt["kbqa_icl"] + state.blocks_state
        if self.prompt['beam_size'] >1:
            outputs_dict = requests.post(self.base_model['simulate'], json={ "input": inputs, "output": ["BEAM",f"{self.prompt['beam_size']}"] }).json()
            outputs = list(outputs_dict.keys())
        else:
            outputs = [requests.post(self.base_model['simulate'], json={ "input": inputs, "output": [] }).json()]
        step_outputs = []
        for output in outputs:
            step_output = '\n'.join(output.split('\n')[:2])+'\n'
            if step_output not in step_outputs:
                step_outputs.append(step_output)
                
        for step_output in step_outputs:
            step_plan = ""
            for plan in plans_dict.keys():
                if step_output.startswith(plans_dict[plan]):
                    step_plan = copy.deepcopy(plan)
                    break
            if step_plan == "":
                continue
            if step_plan == 'Extract_entity':
                candidate_entities = self.get_candidate_entities(state)[:1]
                if len(candidate_entities) == 0:
                    continue
                try:
                    gen_argument = step_output.split('[ ')[1].split(' ]')[0]
                except:
                    continue
                # scores_list = self.retrieval_model.similarity(gen_argument,[e for e,e_id in candidate_entities]).tolist()
                scores_list = [1.0]
                entities_scores_list = [(e,s,e_id) for (e,e_id), s in zip(candidate_entities,scores_list)]
                entities_scores_list = sorted(entities_scores_list, key=lambda x: x[1], reverse=True)
                entities_scores_list = entities_scores_list[:self.prompt['entity_topk']]  
                entities_scores_list = [esl for esl in entities_scores_list if esl[1]>=self.prompt['entity_threshold']]      
                ## Collect Scratchpads
                id_new = str(int(state.expression_id)+1) if state.expression_id != '' else '1' if step_n != 1 else ''
                for entity,_,entity_id in entities_scores_list:
                    scratchpad = plans_dict[step_plan] + f'[ {entity} ]' + f"\nObservation{step_n}: expression{id_new} = START('{entity_id}')\n"
                    func = f"expression{id_new} = START('{entity_id}')"
                    if scratchpad not in [s for s,f in scratchpads_list]:
                        scratchpads_list.append((scratchpad,func))                 
                            
            elif step_plan == 'Find_relation':
                if state.functions_state == []:
                    continue
                candidate_relations = self.get_candidate_relations(state)
                if len(candidate_relations) == 0:
                    continue
                try:
                    gen_argument = step_output.split('[ ')[1].split(' ]')[0]
                except:
                    continue
                scores_list = self.retrieval_model.similarity(gen_argument,[r for r,r_id in candidate_relations]).tolist()
                relations_scores_list = [(r,s,r_id) for (r,r_id), s in zip(candidate_relations,scores_list)]
                relations_scores_list = sorted(relations_scores_list, key=lambda x: x[1], reverse=True)
                relations_scores_list = relations_scores_list[:self.prompt['relation_topk']]
                relations_scores_list = [rsl for rsl in relations_scores_list if rsl[1]>=self.prompt['relation_threshold']]
                ## Collect Scratchpads
                if 'START' in state.functions_state[-1]:
                    ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
                    if ent_type(ent) in ['name','entity','url','onto','int']: 
                        for (relation,_,relation_id) in relations_scores_list:
                            func = f"expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})"
                            scratchpad = plans_dict[step_plan] +  f'[ {relation} ]' + f"\nObservation{step_n}: expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})\n"
                            if scratchpad not in [s for s,f in scratchpads_list]:
                                scratchpads_list.append((scratchpad,func))                    
                    elif ent_type(ent) in ['literal']:
                        for (relation,_,relation_id) in relations_scores_list:
                            func = f"expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})"
                            if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                                scratchpad = plans_dict[step_plan] +  f'[ {relation} ]' + f"\nObservation{step_n}: expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})\n"
                                if scratchpad not in [s for s,f in scratchpads_list]:
                                    scratchpads_list.append((scratchpad,func)) 
                else:
                    for (relation,_,relation_id) in relations_scores_list:
                        func = f"expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})"
                        scratchpad = plans_dict[step_plan] +  f'[ {relation} ]' + f"\nObservation{step_n}: expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})\n"
                        if scratchpad not in [s for s,f in scratchpads_list]:
                            scratchpads_list.append((scratchpad,func))  
                        
            elif step_plan == 'Merge':
                if state.expression_id == '':
                    continue
                id_new = str(int(state.expression_id)-1) if int(state.expression_id) > 1 else ''
                func = f"expression{id_new} = AND(expression{state.expression_id}, expression{id_new})"
                if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                    scratchpad = plans_dict[step_plan] +  f'[ expression{state.expression_id} | expression{id_new} ]' + f"\nObservation{step_n}: expression{id_new} = AND(expression{state.expression_id}, expression{id_new})\n"
                    if scratchpad not in [s for s,f in scratchpads_list]:
                        scratchpads_list.append((scratchpad,func)) 
                    
            elif step_plan == 'Order':
                try:
                    gen_argument = step_output.split('[ ')[1].split(' ]')[0]
                except:
                    continue                           
                candidate_modes = ['ARGMAX','ARGMIN']
                candidate_paths = self.prompt['literal_relation_list']
                candidate_modes_paths_list = [(mode,path) for mode in candidate_modes for path in candidate_paths]
                scores_list = self.retrieval_model.similarity(f'[ {gen_argument} ]',[f'[ {mode} | {path} ]' for mode,path in candidate_modes_paths_list]).tolist()
                modes_paths_scores_list = [(candidate_modes_paths_list[t], s) for t, s in enumerate(scores_list)]
                modes_paths_scores_list = sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)
                modes_paths_scores_list = modes_paths_scores_list[:self.prompt['limit_topk']]
                modes_paths_scores_list = [mpsl for mpsl in modes_paths_scores_list if mpsl[1]>=self.prompt['limit_threshold']]
                ## Collect Scratchpads
                for (mode,path),_ in modes_paths_scores_list:
                    func = f"expression{state.expression_id} = ARG('{mode}', expression{state.expression_id}, '{path}')"
                    if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                        scratchpad = plans_dict[step_plan] +  f'[ {mode} | {path} ]' + f"\nObservation{step_n}: expression{state.expression_id} = ARG('{mode}', expression{state.expression_id}, '{path}')\n"
                        if scratchpad not in [s for s,f in scratchpads_list]:
                            scratchpads_list.append((scratchpad,func)) 
                        
            elif step_plan == 'Compare':
                try:
                    gen_argument = step_output.split('[ ')[1].split(' ]')[0]
                except:
                    continue
                mode_dict = {'le':'LESS EQUAL', 'ge':'GREATER EQUAL', 'lt':'LESS THAN', 'gt':'GREATER THAN'}
                candidate_modes = ['le','ge','lt','gt']
                candidate_paths = self.prompt['literal_relation_list']                            
                candidate_modes_paths_list = [(mode,path) for mode in candidate_modes for path in candidate_paths]
                scores_list = self.retrieval_model.similarity(f'[ {gen_argument} ]',[f'[ {mode_dict[mode]} | {path} ]' for mode,path in candidate_modes_paths_list]).tolist()
                modes_paths_scores_list = [(candidate_modes_paths_list[t], s) for t, s in enumerate(scores_list)]
                modes_paths_scores_list = sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)
                modes_paths_scores_list = modes_paths_scores_list[:self.prompt['limit_topk']]
                modes_paths_scores_list = [mpsl for mpsl in modes_paths_scores_list if mpsl[1]>=self.prompt['limit_threshold']]
                ## Collect Scratchpads
                for (mode,path),_ in modes_paths_scores_list:
                    func = f"expression{state.expression_id} = CMP('{mode}', '{path}', expression{state.expression_id})"
                    if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                        scratchpad = plans_dict[step_plan] +  f'[ {mode_dict[mode]} | {path} ]' + f"\nObservation{step_n}: expression{state.expression_id} = CMP('{mode}', '{path}', expression{state.expression_id})\n"
                        if scratchpad not in [s for s,f in scratchpads_list]:
                            scratchpads_list.append((scratchpad,func)) 
                        
            elif step_plan == 'Time_constraint':
                try:
                    gen_argument = step_output.split('[ ')[1].split(' ]')[0]
                except:
                    continue
                candidate_paths = self.prompt['literal_relation_list'] 
                candidate_times = self.prompt['tc_time_list']
                candidate_paths_times_list = [(path,time) for path in candidate_paths for time in candidate_times]
                scores_list = self.retrieval_model.similarity(f'[ {gen_argument} ]',[f'[ {path} | {time} ]' for path,time in candidate_paths_times_list]).tolist()
                paths_times_scores_list = [(candidate_paths_times_list[t], s) for t, s in enumerate(scores_list)]
                paths_times_scores_list = sorted(paths_times_scores_list, key=lambda x: x[1], reverse=True)  
                paths_times_scores_list = paths_times_scores_list[:self.prompt['limit_topk']]
                paths_times_scores_list = [ptsl for ptsl in paths_times_scores_list if ptsl[1]>=self.prompt['limit_threshold']]
                ## Collect Scratchpads
                for (path,time),_ in paths_times_scores_list:
                    func = f"expression{state.expression_id} = TC(expression{state.expression_id}, '{path}', '{time}')"
                    if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                        scratchpad = plans_dict[step_plan] +  f'[ {path} | {time} ]' + f"\nObservation{step_n}: expression{state.expression_id} = TC(expression{state.expression_id}, '{path}', '{time}')\n"
                        if scratchpad not in [s for s,f in scratchpads_list]:
                            scratchpads_list.append((scratchpad,func))     
                             
            elif step_plan == "Count":
                func = f"expression{state.expression_id} = COUNT(expression{state.expression_id})"
                if len(test_execute_function_list(state.functions_state,self.example['dataset'])) > 0:
                    scratchpad = plans_dict[step_plan] +  f'[ expression{state.expression_id} ]' + f"\nObservation{step_n}: expression{state.expression_id} = COUNT(expression{state.expression_id})\n"
                    if scratchpad not in [s for s,f in scratchpads_list]:
                        scratchpads_list.append((scratchpad,func))                     
                    
            elif step_plan == 'Finish':
                func = f"expression{state.expression_id} = STOP(expression{state.expression_id})"
                if len(test_execute_function_list(state.functions_state,self.example['dataset'])) > 0:
                    scratchpad = plans_dict[step_plan] +  f'[ expression{state.expression_id} ]' + f"\nObservation{step_n}: expression{state.expression_id} = STOP(expression{state.expression_id})\n"
                    if scratchpad not in [s for s,f in scratchpads_list]:
                        scratchpads_list.append((scratchpad,func)) 
        
        scratchpads_scores_list = []          
        for scratchpad,func in scratchpads_list:
            if "= STOP" in scratchpad:
                simulate_output = ""
            else:
                simulate_output = requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"] + state.blocks_state + scratchpad, "output": []}).json()
            simulate_output = state.blocks_state + scratchpad + simulate_output
            score = requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"], "output": [simulate_output]}).json()[0]
            # score = requests.post(self.base_model['reward'], json={ "input": self.example["question"], "output": [simulate_sexpr]}).json()[0]
            scratchpads_scores_list.append((scratchpad,score,func)) 
        scratchpads_scores_list = [(a,min(r,100.0),f) for a,r,f in scratchpads_scores_list]
        scratchpads_scores_list = sorted(scratchpads_scores_list, key=lambda x: x[1], reverse=True)
        scratchpads_scores_list = scratchpads_scores_list[:self.prompt['step_topk']]
        scratchpads_scores_list = [ssl for ssl in scratchpads_scores_list if ssl[1]>=self.prompt['step_threshold']]
        if scratchpads_scores_list == []:
            func = f"expression{state.expression_id} = STOP(expression{state.expression_id})"         
            scratchpad = plans_dict['Finish'] + f'[ expression{state.expression_id} ]' + f"\nObservation{step_n}: expression{state.expression_id} = STOP(expression{state.expression_id})\n"
            panel_score = self.prompt['noway_panelty']
            scratchpads_scores_list = [(scratchpad, panel_score, func)]
        return scratchpads_scores_list        
        
    def get_a(self, state: AgentState) -> list[AgentAction]:     
        scratchpads_list = []
        step_n = state.step_idx + 1   
        # Select Plans
        ## Select Plans with rules
        plans_dict = {plan: PLAN_LIST[plan].format(step_n) for plan in PLAN_LIST.keys()}
        plans = list(plans_dict.values())
        plans = list(set(plans) - set([plans_dict['Count']])) 
 
        ### Rules when given candidate entities
        candidate_entities = copy.deepcopy(self.example['entities'])
        candidate_tc_time = []
        for ent,ent_id in candidate_entities:
            if ent_type(ent_id) == "int":
                if ent in self.prompt['tc_time_list']:
                    candidate_entities.remove((ent,ent_id))
                    candidate_tc_time.append((ent,ent_id))
        for ent_id in re.findall(f"START\(\'(.*?)\'\)", '\n'.join(state.functions_state)):
            for ent in candidate_entities:
                if ent_id == ent[1]:
                    candidate_entities.remove(ent)
                    break
        if len(candidate_entities) == 0:
            plans = list(set(plans) - set([plans_dict['Extract_entity']]))
        else:
            plans = list(set(plans) - set([plans_dict['Finish']]))
                       
        ### Rules of Extract_entity
        if step_n == 1:
            plans = list(set([plans_dict['Extract_entity']]))
        else:
            # Previous step limitation
            if "= JOIN" in state.functions_state[-1]:
                pass   
            elif "= AND" in state.functions_state[-1]:
                pass 
            elif "= CMP" in state.functions_state[-1]:
                pass   
            else:
                plans = list(set(plans) - set([plans_dict['Extract_entity']]))                        
        
        ### Rules of Find_relation
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Find_relation']]))
        else:
            # Previous step limitation
            if "= JOIN" in state.functions_state[-1]:
                pass
            elif "= START" in state.functions_state[-1] and ent_type(re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]) in ['entity','name','onto','literal','url']:
                pass  
            elif '= TC' in state.functions_state[-1]:
                pass 
            elif '= AND' in state.functions_state[-1]:
                pass
            elif '= ARG' in state.functions_state[-1]:
                pass
            elif '= CMP' in state.functions_state[-1]:
                pass
            else:
                plans = list(set(plans) - set([plans_dict['Find_relation']]))
            # Next step limitation
            if '= JOIN' in state.functions_state[-1]: 
                plans = list(set(plans) - set([plans_dict['Count'],plans_dict['Compare']]))        
             
        ### Rules of Merge
        if state.expression_id == '':
            plans = list(set(plans) - set([plans_dict['Merge']]))  
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Merge']]))
        else:
            # Previous step limitation
            if "= JOIN" in state.functions_state[-1]:
                if "JOIN('people.person.gender'," in state.functions_state[-1]:
                    plans = list(set([plans_dict['Merge']])) 
                    if state.expression_id == '':
                        plans = []
                if "JOIN('common.topic.notable_types'," in state.functions_state[-1]:
                    plans = list(set([plans_dict['Merge']])) 
                    if state.expression_id == '':
                        plans = []
                pass
            elif "= CMP" in state.functions_state[-1]:
                pass
            elif "= START" in state.functions_state[-1] and ent_type(re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]) in ['onto']:
                pass 
            else:
                plans = list(set(plans) - set([plans_dict['Merge']]))
            # Next step limitation
            if '= AND' in state.functions_state[-1]: 
                plans = list(set(plans) - set([plans_dict['Compare'],plans_dict['Merge']]))
                     
        ### Rules of Order     
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Order']]))
        else:
            # Previous step limitation
            if '= JOIN' in state.functions_state[-1]: 
                pass
            elif '= AND' in state.functions_state[-1]: 
                pass
            elif '= START' in state.functions_state[-1] and ent_type(re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]) == 'onto':
                pass
            else:
                plans = list(set(plans) - set([plans_dict['Order']]))
            # Next step limitation
            if '= ARG' in state.functions_state[-1]: 
                plans = list(set(plans) - set([plans_dict['Extract_entity'],plans_dict['Merge'],plans_dict['Order'],plans_dict['Compare'],plans_dict['Time_constraint'],plans_dict['Count']]))
             
        ### Rules of Compare    
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Compare']]))
        else:
            # Previous step limitation
            if '= START' in state.functions_state[-1] and ent_type(re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]) in ['literal','int']:
                pass
            else:
                plans = list(set(plans) - set([plans_dict['Compare']]))
            # Next step limitation  
            if '= CMP' in state.functions_state[-1]: 
                plans = list(set(plans) - set([plans_dict['Order'],plans_dict['Compare'],plans_dict['Time_constraint'],plans_dict['Count'],plans_dict['Finish']]))
   
        ### Rules of Time_constraint   
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Time_constraint']]))
        else:
            # Previous step limitation
            if "= JOIN" in state.functions_state[-1]:
                pass   
            elif "= AND" in state.functions_state[-1]:
                pass 
            else:
                plans = list(set(plans) - set([plans_dict['Time_constraint']]))
            # Next step limitation  
            if '= TC' in state.functions_state[-1]: 
                plans = list(set(plans) - set([plans_dict['Extract_entity'],plans_dict['Merge'],plans_dict['Order'],plans_dict['Compare'],plans_dict['Time_constraint'],plans_dict['Count']]))
   
        ### Rules of Count   
        if state.expression_id != '':
            plans = list(set(plans) - set([plans_dict['Count']]))
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Count']]))
        else:
            # Previous step limitation
            if "= AND" in state.functions_state[-1]:
                pass   
            else:
                plans = list(set(plans) - set([plans_dict['Count']]))
            # Next step limitation  
            if '= COUNT' in state.functions_state[-1]: 
                plans = list(set([plans_dict['Finish']]))

        ### Rules of Finish
        if step_n == self.prompt['deapth_limit']:
            plans = [plans_dict['Finish']]
        if state.expression_id != '':
            plans = list(set(plans) - set([plans_dict['Finish']]))
        if step_n == 1:
            plans = list(set(plans) - set([plans_dict['Finish']]))
        else:
            # Previous step limitation
            if "= AND" in state.functions_state[-1]:
                pass
            elif "= JOIN" in state.functions_state[-1]:
                pass
            elif "= ARG" in state.functions_state[-1]:
                pass        
            elif "= TC" in state.functions_state[-1]:
                pass     
            elif "= COUNT" in state.functions_state[-1]:
                pass  
            else:
                plans = list(set(plans) - set([plans_dict['Finish']])) 
                
        if "= TC" in state.blocks_state:
            plans = list(set(plans) - set([plans_dict['Time_constraint']])) 
            if self.example['dataset'] == 'WebQSP':
                 plans = list(set(plans) - set([plans_dict['Order']]))
        if "= ARG" in state.blocks_state:
            plans = list(set(plans) - set([plans_dict['Order']]))
            if self.example['dataset'] == 'WebQSP':
                plans = list(set(plans) - set([plans_dict['Time_constraint']])) 
            elif self.example['dataset'] == 'CWQ':
                plans = list(set(plans) - set([plans_dict['Compare']]))
        if "= CMP" in state.blocks_state:
            plans = list(set(plans) - set([plans_dict['Compare']])) 
            plans = list(set(plans) - set([plans_dict['Order']]))
            
        if '= TC' not in self.example['function_prompt']:
            plans = list(set(plans) - set([plans_dict['Time_constraint']]))
        else:
            if '= TC' not in state.blocks_state:
                plans = list(set(plans) - set([plans_dict['Finish']]))
        if '= CMP' not in self.example['function_prompt']:
            plans = list(set(plans) - set([plans_dict['Compare']]))
        else:
            if '= CMP' not in state.blocks_state:
                plans = list(set(plans) - set([plans_dict['Finish']]))
        if '= ARG' not in self.example['function_prompt']:
            plans = list(set(plans) - set([plans_dict['Order']]))
        else:
            if '= ARG' not in state.blocks_state:
                plans = list(set(plans) - set([plans_dict['Finish']]))
                
        if 'type.object.name' in state.buffered_action:
            plans = [plans_dict['Find_relation']] if self.example['dataset'] == 'CWQ' else [plans_dict['Finish']]
        
        if step_n >=2:
            if '= START' in state.functions_state[-1]:
                if re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1] in ['m.01y2hnl', 'm.01mp', 'm.0kpys4', 'm.01nt', 'm.01xpjyz', 'm.0kpym_', 'm.02h76fz', 'm.01xryvm', 'm.01xljv1', 'm.044801x', 'm.01y2hn6', 'm.01mh', 'm.04fnrhx']:
                    plans = [plans_dict['Find_relation']]
            if '\'common.topic.notable_types\'' in state.functions_state[-1]:
                plans = list(set(plans) & set([plans_dict['Merge']]))
                    
        ## Select Plans with scores
        inputs = self.prompt["kbqa_icl"] + state.blocks_state
        if len(plans) > self.prompt['plan_topk']:
            plans_scores_list = [(plans[t],score) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": inputs, "output": ["Thought{}: ".format(step_n) + plan for plan in plans] }).json())]
            plans_scores_list = sorted(plans_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['plan_topk']]
        else:
            plans_scores_list = [(plan,100.0) for plan in plans]
        
        if self.prompt['ex']:   
            if self.example['function_prompt'].startswith(state.blocks_state):
                plan = self.example['function_prompt'].split(f'Action{step_n}: ')[-1].split(' [')[0]
                plans_scores_list = [(plans_dict[plan],100.0)] + [p for p in plans_scores_list if p[0] != plans_dict[plan]]
                

   
        # Select Arguments
        for plan,_ in plans_scores_list:
            if "Extract_entity" in plan:
                ## Select START Entity
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan 
                if len(candidate_entities) > self.prompt['start_entity_topk']:
                    entities_scores_list = [(candidate_entities[t][0],score,candidate_entities[t][1]) for t, score in enumerate(requests.post(self.base_model['select'], json={ "input": inputs, "output": [f'[ {ent} ]' for ent,_ in candidate_entities] }).json())]              
                    entities_scores_list = sorted(entities_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['start_entity_topk']] 
                else:
                    entities_scores_list = [(ent[0],100.0,ent[1]) for ent in candidate_entities]
                
                if self.prompt['ex']:   
                    if (self.prompt['kbqa_icl']+self.example['function_prompt']).startswith(inputs):
                        golden_argument = '['+self.example['function_prompt'].split(f'Action{step_n}: ')[-1].split(f'[')[1].split(']')[0]+']'
                        for item in entities_scores_list[:self.prompt['ex_topk']]:
                            candidate_argument = '[ ' + item[0]+ ' ]'
                            if golden_argument != candidate_argument:
                                self.example['select_dpo'].append({"conversations": 
                                    [{"from": "human", "value": inputs}],
                                    "chosen": {"from": "gpt", "value": golden_argument},
                                    "rejected": {"from": "gpt","value": candidate_argument}}) 
                                    
                entities_scores_list = entities_scores_list[:self.prompt['start_entity_topk']]        
                ## Collect Scratchpads
                id_new = str(int(state.expression_id)+1) if state.expression_id != '' else '1' if step_n != 1 else ''
                for entity,_,entity_id in entities_scores_list:
                    scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {entity} ]' + f"\nObservation{step_n}: expression{id_new} = START('{entity_id}')\n"
                    func = f"expression{id_new} = START('{entity_id}')"
                    scratchpads_list.append((scratchpad,func))  

            elif "Find_relation" in plan:
                ## Select JOIN Relation      
                if 'START' in state.functions_state[-1]:
                    ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
                    if ent_type(ent) in ['name', 'literal', 'int']:
                        candidate_r_relations = []
                        candidate_relations = []
                        for rel in (self.prompt['literal_relation_list'] if ent_type(ent) != 'name' else self.prompt['name_relation_list']):
                            if 'JOIN' not in rel:
                                if '(R ' in rel:
                                    candidate_r_relations.append(re.findall(f"\(R (.*?)\)", rel)[0])
                                else:
                                    candidate_relations.append(rel)
                    else:
                        candidate_r_relations = get_next_r_relations(state.functions_state,self.example['dataset'])
                        candidate_relations = get_next_relations(state.functions_state,self.example['dataset'])
                else:
                    candidate_r_relations = get_next_r_relations(state.functions_state,self.example['dataset'])
                    candidate_relations = get_next_relations(state.functions_state,self.example['dataset'])
                
                
                candidate_relations = [(rel,f'(R {rel})') for rel in candidate_r_relations if rel not in self.prompt['join_ban_relation_list'] and rel in self.prompt['relation_list']] + [(rel,f'{rel}') for rel in candidate_relations if rel not in self.prompt['join_ban_relation_list'] and rel in self.prompt['relation_list']]
                if step_n >=2:
                    if '= START' in state.functions_state[-1]:
                        if re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1] in ['m.01y2hnl', 'm.01mp', 'm.0kpys4', 'm.01nt', 'm.01xpjyz', 'm.0kpym_', 'm.02h76fz', 'm.01xryvm', 'm.01xljv1', 'm.044801x', 'm.01y2hn6', 'm.01mh', 'm.04fnrhx']:
                            candidate_relations = [('common.topic.notable_types','common.topic.notable_types')]               
                
                if self.example['dataset'] == 'WebQSP':
                    if 'START' in state.functions_state[-1]:
                        ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
                        if ent_type(ent) != 'entity':   
                            candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if rel!='type.object.name']
                    else:
                        candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if rel!='type.object.name']
                if self.example['dataset'] == 'CWQ':
                    if 'START' in state.functions_state[-1]:
                        ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
                        if not (ent_type(ent) in ['name'] and ent.endswith("@en")):   
                            candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if rel!='type.object.name'] 
                    else:
                        candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if rel!='type.object.name']   
                
                if self.prompt['ex']:
                    candidate_relations = [(rel,rel_id) for rel,rel_id in candidate_relations if not (rel_id in state.blocks_state)]
                
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
                if len(candidate_relations) > self.prompt['join_retrieval_topk']:
                    generated_relation = requests.post(self.base_model['select'], json={ "input": inputs, "output": [] }).json()
                    relations_scores_list = [(candidate_relations[t][0], s, candidate_relations[t][1]) for t, s in enumerate(self.retrieval_model.similarity(generated_relation,[f'[ {r[0]} ]' for r in candidate_relations]).tolist())]
                    relations_scores_list = [rel for rel in sorted(relations_scores_list, key=lambda x: x[1], reverse=True)][:self.prompt['join_retrieval_topk']]
                else:
                    relations_scores_list = [(rel[0],100.0,rel[1]) for rel in candidate_relations]
                
                if len(relations_scores_list) > self.prompt['join_relation_topk']:
                    relations_scores_list = [(relations_scores_list[t][0],score,relations_scores_list[t][2]) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"] + state.blocks_state, "output": ["Thought{}: ".format(step_n) + plan+f'[ {rel[0]} ]' for rel in relations_scores_list] }).json())]
                    relations_scores_list = sorted(relations_scores_list, key=lambda x: x[1], reverse=True)

                if self.prompt['ex']:   
                    if (self.prompt['kbqa_icl']+self.example['function_prompt']).startswith(inputs):
                        golden_argument = '['+self.example['function_prompt'].split(f'Action{step_n}: ')[-1].split(f'[')[1].split(']')[0]+']'
                        for item in relations_scores_list[:self.prompt['ex_topk']]:
                            candidate_argument = '[ ' + item[0]+ ' ]'
                            if golden_argument != candidate_argument:
                                self.example['select_dpo'].append({"conversations": 
                                    [{"from": "human", "value": inputs}],
                                    "chosen": {"from": "gpt", "value": golden_argument},
                                    "rejected": {"from": "gpt","value": candidate_argument}}) 
                
                relations_scores_list = relations_scores_list[:self.prompt['join_relation_topk']]
                ## Collect Scratchpads
                if 'START' in state.functions_state[-1]:
                    ent = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", state.functions_state[-1])[0][1]
                    if ent_type(ent) in ['name','entity','url','onto','int']: 
                        for (relation,_,relation_id) in relations_scores_list:
                            func = f"expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})"
                            scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {relation} ]' + f"\nObservation{step_n}: expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})\n"
                            scratchpads_list.append((scratchpad,func))                    
                    elif ent_type(ent) in ['literal']:
                        for (relation,_,relation_id) in relations_scores_list:
                            func = f"expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})"
                            if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                                scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {relation} ]' + f"\nObservation{step_n}: expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})\n"
                                scratchpads_list.append((scratchpad,func)) 
                else:
                    for (relation,_,relation_id) in relations_scores_list:
                        func = f"expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})"
                        scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {relation} ]' + f"\nObservation{step_n}: expression{state.expression_id} = JOIN('{relation_id}', expression{state.expression_id})\n"
                        scratchpads_list.append((scratchpad,func))

            elif "Merge" in plan:
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
                id_new = str(int(state.expression_id)-1) if int(state.expression_id) > 1 else ''
                func = f"expression{id_new} = AND(expression{state.expression_id}, expression{id_new})"
                if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                    scratchpad = "Thought{}: ".format(step_n) + plan + f'[ expression{state.expression_id} | expression{id_new} ]' + f"\nObservation{step_n}: expression{id_new} = AND(expression{state.expression_id}, expression{id_new})\n"
                    scratchpads_list.append((scratchpad,func)) 

            elif "Order" in plan:
                candidate_modes = ['ARGMAX','ARGMIN']
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
                
                generated_mode_path = requests.post(self.base_model['select'], json={ "input": inputs, "output": [] }).json()
                candidate_modes_paths_list = [(mode,path) for mode in candidate_modes for path in self.prompt['literal_relation_list']]
                modes_paths_scores_list = [(candidate_modes_paths_list[t], s) for t, s in enumerate(self.retrieval_model.similarity(generated_mode_path,[f'[ {mode} | {path} ]' for mode,path in candidate_modes_paths_list]).tolist())]
                
                # modes_paths_scores_list = [mp for mp in sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)][:self.prompt['limit_retrieval_topk']]
                modes_paths_scores_list = [mp for mp in sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)]
                if len(modes_paths_scores_list) > self.prompt['limit_retrieval_topk']:
                    modes_paths_scores_list = [(modes_paths_scores_list[t][0],score) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"] + state.blocks_state, "output": ["Thought{}: ".format(step_n) + plan+f'[ {mode} | {path} ]' for (mode,path),_ in modes_paths_scores_list] }).json())]
                    modes_paths_scores_list = sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)
                    
                if self.prompt['ex']:   
                    if (self.prompt['kbqa_icl']+self.example['function_prompt']).startswith(inputs):
                        golden_argument = '['+self.example['function_prompt'].split(f'Action{step_n}: ')[-1].split(f'[')[1].split(']')[0]+']'
                        for item in modes_paths_scores_list[:self.prompt['ex_topk']]:
                            candidate_argument = '[ ' + item[0][0] + ' | ' + item[0][1] + ' ]'
                            if golden_argument != candidate_argument:
                                self.example['select_dpo'].append({"conversations": 
                                    [{"from": "human", "value": inputs}],
                                    "chosen": {"from": "gpt", "value": golden_argument},
                                    "rejected": {"from": "gpt","value": candidate_argument}}) 
                
                modes_paths_scores_list = modes_paths_scores_list[:self.prompt['limit_retrieval_topk']]  
                for (mode,path),_ in modes_paths_scores_list:
                    func = f"expression{state.expression_id} = ARG('{mode}', expression{state.expression_id}, '{path}')"
                    if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                        scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {mode} | {path} ]' + f"\nObservation{step_n}: expression{state.expression_id} = ARG('{mode}', expression{state.expression_id}, '{path}')\n"
                        scratchpads_list.append((scratchpad,func))                     
                 
            elif "Compare" in plan:
                mode_dict = {'le':'LESS EQUAL', 'ge':'GREATER EQUAL', 'lt':'LESS THAN', 'gt':'GREATER THAN'}
                candidate_modes = ['le','ge','lt','gt']
                candidate_paths = self.prompt['literal_relation_list']
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
                generated_mode_path = requests.post(self.base_model['select'], json={ "input": inputs, "output": [] }).json()
                candidate_modes_paths_list = [(mode,path) for mode in candidate_modes for path in candidate_paths]
                modes_paths_scores_list = [(candidate_modes_paths_list[t], s) for t, s in enumerate(self.retrieval_model.similarity(generated_mode_path,[f'[ {mode_dict[mode]} | {path} ]' for mode,path in candidate_modes_paths_list]).tolist())]
                
                # modes_paths_scores_list = [mp for mp in sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)][:self.prompt['limit_retrieval_topk']]
                modes_paths_scores_list = [mp for mp in sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)]
                if len(modes_paths_scores_list) > self.prompt['limit_retrieval_topk']:
                    modes_paths_scores_list = [(modes_paths_scores_list[t][0],score) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"] + state.blocks_state, "output": ["Thought{}: ".format(step_n) + plan+f'[ {mode_dict[mode]} | {path} ]' for (mode,path),_ in modes_paths_scores_list] }).json())]
                    modes_paths_scores_list = sorted(modes_paths_scores_list, key=lambda x: x[1], reverse=True)   
                    
                if self.prompt['ex']:   
                    if (self.prompt['kbqa_icl']+self.example['function_prompt']).startswith(inputs):
                        golden_argument = '['+self.example['function_prompt'].split(f'Action{step_n}: ')[-1].split(f'[')[1].split(']')[0]+']'
                        for item in modes_paths_scores_list[:self.prompt['ex_topk']]:
                            candidate_argument = '[ ' + item[0][0] + ' | ' + item[0][1] + ' ]'
                            if golden_argument != candidate_argument:
                                self.example['select_dpo'].append({"conversations": 
                                    [{"from": "human", "value": inputs}],
                                    "chosen": {"from": "gpt", "value": golden_argument},
                                    "rejected": {"from": "gpt","value": candidate_argument}})             
                
                modes_paths_scores_list = modes_paths_scores_list[:self.prompt['limit_retrieval_topk']] 
                for (mode,path),_ in modes_paths_scores_list:
                    func = f"expression{state.expression_id} = CMP('{mode}', '{path}', expression{state.expression_id})"
                    if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                        scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {mode_dict[mode]} | {path} ]' + f"\nObservation{step_n}: expression{state.expression_id} = CMP('{mode}', '{path}', expression{state.expression_id})\n"
                        scratchpads_list.append((scratchpad,func))                 
                            
            elif "Time_constraint" in plan:
                ## Select TC Time
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
                candidate_paths = self.prompt['literal_relation_list']
                
                if len(candidate_tc_time) == 1:
                    times_list = [time for time,_ in candidate_tc_time]
                else:
                    times_list = ['NOW']
                generated_path_time = requests.post(self.base_model['select'], json={ "input": inputs, "output": [] }).json()
                candidate_paths_times_list = [(path,time) for path in candidate_paths for time in times_list]
                paths_times_scores_list = [(candidate_paths_times_list[t], s) for t, s in enumerate(self.retrieval_model.similarity(generated_path_time,[f'[ {path} | {time} ]' for path,time in candidate_paths_times_list]).tolist())]
                
                # paths_times_scores_list = [pt for pt in sorted(paths_times_scores_list, key=lambda x: x[1], reverse=True)][:self.prompt['limit_retrieval_topk']]
                paths_times_scores_list = [pt for pt in sorted(paths_times_scores_list, key=lambda x: x[1], reverse=True)]
                if len(paths_times_scores_list) > self.prompt['limit_retrieval_topk']:
                    paths_times_scores_list = [(paths_times_scores_list[t][0],score) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"] + state.blocks_state, "output": ["Thought{}: ".format(step_n) + plan+f'[ {path} | {time} ]' for (path,time),_ in paths_times_scores_list] }).json())]
                    paths_times_scores_list = sorted(paths_times_scores_list, key=lambda x: x[1], reverse=True)  
                    
                if self.prompt['ex']:   
                    if (self.prompt['kbqa_icl']+self.example['function_prompt']).startswith(inputs):
                        golden_argument = '['+self.example['function_prompt'].split(f'Action{step_n}: ')[-1].split(f'[')[1].split(']')[0]+']'
                        for item in paths_times_scores_list[:self.prompt['ex_topk']] :
                            candidate_argument = '[ ' + item[0][0] + ' | ' + item[0][1] + ' ]'
                            if golden_argument != candidate_argument:
                                self.example['select_dpo'].append({"conversations": 
                                    [{"from": "human", "value": inputs}],
                                    "chosen": {"from": "gpt", "value": golden_argument},
                                    "rejected": {"from": "gpt","value": candidate_argument}})             
                
                paths_times_scores_list = paths_times_scores_list[:self.prompt['limit_retrieval_topk']]   
                for (path,time),_ in paths_times_scores_list:
                    func = f"expression{state.expression_id} = TC(expression{state.expression_id}, '{path}', '{time}')"
                    if len(test_execute_function_list(state.functions_state+[func],self.example['dataset'])) > 0:
                        scratchpad = "Thought{}: ".format(step_n) + plan + f'[ {path} | {time} ]' + f"\nObservation{step_n}: expression{state.expression_id} = TC(expression{state.expression_id}, '{path}', '{time}')\n"
                        scratchpads_list.append((scratchpad,func))
     
            # elif "Count" in plan:
            #     inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
            #     func = f"expression{state.expression_id} = COUNT(expression{state.expression_id})"
            #     if len(test_execute_function_list(state.functions_state,self.example['dataset'])) > 0:
            #         scratchpad = "Thought{}: ".format(step_n) + plan + f'[ expression{state.expression_id} ]' + f"\nObservation{step_n}: expression{state.expression_id} = COUNT(expression{state.expression_id})\n"
            #         scratchpads_list.append((scratchpad,func)) 
          
            elif "Finish" in plan:
                inputs = self.prompt["kbqa_icl"] + state.blocks_state + "Thought{}: ".format(step_n) + plan
                func = f"expression{state.expression_id} = STOP(expression{state.expression_id})"
                if len(test_execute_function_list(state.functions_state,self.example['dataset'])) > 0:
                    scratchpad = "Thought{}: ".format(step_n) + plan + f'[ expression{state.expression_id} ]' + f"\nObservation{step_n}: expression{state.expression_id} = STOP(expression{state.expression_id})\n"
                    scratchpads_list.append((scratchpad,func)) 
        
        inputs = copy.deepcopy(self.prompt["kbqa_icl"])
        if scratchpads_list != []:
            
            threshold = copy.deepcopy(self.prompt['reflect_threshold'])
            if self.prompt['ex']:
                if self.example['function_prompt'].startswith(state.blocks_state):
                    threshold = copy.deepcopy(self.prompt['ex_threshold'])
            
            if self.example['dataset'] == "WebQSP":
                scratchpads_scores_list = [(scratchpads_list[t][0],score,scratchpads_list[t][1]) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": inputs + state.blocks_state, "output": [scratchpad for scratchpad,_ in scratchpads_list]}).json())]
                # scratchpads_scores_list = [(scratchpads_list[t][0],score,scratchpads_list[t][1]) for t, score in enumerate(requests.post(self.base_model['reward'], json={ "input": inputs, "output": [state.blocks_state + scratchpad for scratchpad,_ in scratchpads_list]}).json())]
                scratchpads_scores_list = [(scratchpads_scores[0],scratchpads_scores[1]+10.0,scratchpads_scores[2]) if '= START' in scratchpads_scores[2] else scratchpads_scores for scratchpads_scores in scratchpads_scores_list]
                scratchpads_scores_list = sorted(scratchpads_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['reflect_topk']]
                scratchpads_scores_list = [scratchpads_scores for scratchpads_scores in scratchpads_scores_list if scratchpads_scores[1] >= threshold or '= START' in scratchpads_scores[2] or ((not self.prompt['ex']) and ('= ARG' in scratchpads_scores[2] or '= CMP' in scratchpads_scores[2] or '= TC' in scratchpads_scores[2] or '= STOP' in scratchpads_scores[2] or ('= JOIN' in scratchpads_scores[2] and len(plans)==1 and scratchpads_scores[2]==scratchpads_scores_list[0][2] and scratchpads_scores[1] < threshold)))]
            elif self.example['dataset'] == "CWQ":
                scratchpads_scores_list = [(scratchpads_list[t][0],score,scratchpads_list[t][1]) for t, score in enumerate(requests.post(self.base_model['simulate'], json={ "input": inputs + state.blocks_state, "output": [scratchpad for scratchpad,_ in scratchpads_list]}).json())] 
                scratchpads_scores_list = [scratchpads_scores for scratchpads_scores in scratchpads_scores_list if scratchpads_scores[1] >= threshold or '= START' in scratchpads_scores[2] or ((not self.prompt['ex']) and ('= ARG' in scratchpads_scores[2] or '= CMP' in scratchpads_scores[2] or '= TC' in scratchpads_scores[2] or '= STOP' in scratchpads_scores[2] or ('= JOIN' in scratchpads_scores[2] and len(plans)==1 and scratchpads_scores[2]==scratchpads_scores_list[0][2] and scratchpads_scores[1] < threshold)))] 
                scratchpads_scores_list = [(scratchpads_scores[0],scratchpads_scores[1]+10.0,scratchpads_scores[2]) if '= START' in scratchpads_scores[2] else scratchpads_scores for scratchpads_scores in scratchpads_scores_list]
                scratchpads_scores_list = sorted(scratchpads_scores_list, key=lambda x: x[1], reverse=True)[:self.prompt['reflect_topk']]
                 
        else:
            scratchpads_scores_list = []   
        
        if scratchpads_scores_list == []:
            func = f"expression{state.expression_id} = STOP(expression{state.expression_id})"         
            scratchpad = "Thought{}: ".format(step_n) + plans_dict['Finish'] + f'[ expression{state.expression_id} ]' + f"\nObservation{step_n}: expression{state.expression_id} = STOP(expression{state.expression_id})\n"
            panel_score = requests.post(self.base_model['simulate'], json={ "input": self.prompt["kbqa_icl"] + state.blocks_state , "output": [scratchpad]}).json()[0] - self.prompt['noway_panelty']
            scratchpads_scores_list = [(scratchpad, panel_score, func)]
            
        scratchpads_scores_list = [(a,min(r,100.0),f) for a,r,f in scratchpads_scores_list]
           
        if self.prompt['ex']: 
            if self.example['function_prompt'].startswith(state.blocks_state):
                gt_action = '\n'.join(self.example['function_prompt'][len(state.blocks_state):].split("\n")[:3])+'\n'
                gt_func = self.example['function_list'][step_n-1]
                scratchpads_scores_list = [(gt_action, 100.1, gt_func)]+[(a,r,f) for a,r,f in scratchpads_scores_list if a!=gt_action or f!=gt_func]

        return scratchpads_scores_list

    def fast_reward(self, state: AgentState, action: AgentAction) -> tuple[float, dict]:
        # if state.buffered_action == "":
        #     current_blocks_state = state.blocks_state
        # else:
        #     current_blocks_state = state.last_blocks_state
        # previous_action = state.buffered_action if state.buffered_action != "" else ""
        
        # every two steps, we will also reduce the icl examples by 2 steps
        # so that the distribution of step length in examples is more reasonable
        # icl_template = self.prompt["icl_list"][state.step_idx // 2]
        
        # inputs = (icl_template.replace("<init_state>", current_blocks_state)
        #                       .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
        #                       .replace("<action>", previous_action))
        # intuition = requests.post(self.base_model, json={ "input": inputs, "output": inputs + action }).json()['Score']
        
        # inputs = self.prompt["kbqa_icl"] + state.last_blocks_state + "Thought {}: ".format(state.step_idx)
        # intuition = requests.post(self.base_model, json={ "input": inputs, "output": action }).json()['Score']

        # self_eval_prompt = (self.prompt["self-eval"]
        #                         .replace("<init_state>", current_blocks_state)
        #                         .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
        #                         .replace("<action>", action))
        # self_eval = requests.post(self.base_model, json={ "input": self_eval_prompt, "output": self_eval_prompt + "good" }).json()['Score']
        
        intuition = action[1]
        self_eval = intuition

        return (self.calculate_reward(intuition, self_eval),
                {'intuition': intuition, "self_eval": self_eval})

    def calculate_reward(self, intuition, goal_reached=None) -> float:
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = goal_reached[1]
        else:
            goal_reward = goal_reached[1]
        return intuition * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: AgentState, action: AgentAction,
               intuition: float = None) -> tuple[float, dict]:
        if "= STOP" in action[1]:
            goal_reached_if = True
            sexpr = get_sexpr(state.blocks_state,self.example['entities'])
            goal_reached_score = requests.post(self.base_model['reward'], json={ "input": self.example['question'], "output": [sexpr]}).json()[0]
            goal_reached = (goal_reached_if, goal_reached_score)
        else:
            goal_reached = (False, 0.0)
        return (self.calculate_reward(intuition, goal_reached),
                {'intuition': intuition, 'goal_reached': goal_reached})

from reasoners.visualization import visualize,visualize_save,visualize_out
from reasoners.visualization.tree_snapshot import NodeData, EdgeData
from reasoners.algorithm.mcts import MCTSNode

# (Optional) You can write node_data_factory and edge_data_factory to show customized information.
def blocksworld_node_data_factory(n: MCTSNode) -> NodeData:
    return NodeData({"block state": n.state.blocks_state if n.state else "Not expanded",
                    #  "function state": '\n'.join(n.state.functions_state) if n.state else "Not expanded",
                    #  "# goals satisfied": n.reward_details["goal_reached"][1] if hasattr(n, "reward_details") else "N/A",
                     "Q": n.Q,
                     "intuition": n.fast_reward_details["intuition"] if n.id!=0 else "N/A",
                     "# visited": len(n.cum_rewards)})

def blocksworld_edge_data_factory(n: MCTSNode) -> EdgeData:
    return EdgeData({# "Q": n.Q,
                    #  "intuition": n.fast_reward_details["intuition"],
                    #  "self_eval": n.fast_reward_details["intuition"],
                     "action": n.action})
    
def visualize_mcts(result_rap):
    visualize(result_rap,
            node_data_factory=blocksworld_node_data_factory,
            edge_data_factory=blocksworld_edge_data_factory)   
    
def visualize_mcts_save(result_rap):
    return visualize_save(result_rap,
            node_data_factory=blocksworld_node_data_factory,
            edge_data_factory=blocksworld_edge_data_factory)  
    
def visualize_mcts_out(data):
    visualize_out(data) 

