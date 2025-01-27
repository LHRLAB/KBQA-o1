import argparse
from tqdm import tqdm
import os
from utils.components.utils import dump_json
from utils.executor.sparql_executor import execute_query_with_odbc
from utils.executor.logic_form_util import lisp_to_nested_expression
from utils.database import functions_to_expression

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WebQSP', help='dataset')
    return parser.parse_args()

def to_agent(split_sexpr, id_now=''): 
    START = "expression{} = START('{}')"
    JOIN = "expression{} = JOIN('{}', {})"
    AND = "expression{} = AND({}, {})"
    ARG = "expression{} = ARG('{}', {}, '{}')"
    CMP = "expression{} = CMP('{}', '{}', {})"
    TC = "expression{} = TC({}, '{}', '{}')"
    COUNT = "expression{} = COUNT({})"
    STOP = "expression = STOP(expression)"
    agent_sexpr = []    
    function_list = []              
    if split_sexpr[0]=='JOIN':
        if len(split_sexpr) > 3 and split_sexpr[2][0]=='"':
            split_sexpr = ['JOIN', split_sexpr[1], ' '.join(split_sexpr[2:])]
        if split_sexpr[1][0] == 'R':
            split_sexpr[1] = f'(R {split_sexpr[1][1]})'
        if isinstance(split_sexpr[2],list):
            temp_sexpr, temp_func = to_agent(split_sexpr[2], id_now)
            agent_sexpr += temp_sexpr
            function_list += temp_func
            split_sexpr[2] = f'{agent_sexpr[-1]}'
        else:
            if split_sexpr[2].startswith('m.') or split_sexpr[2].startswith('g.'):
                agent_sexpr.append(f'{split_sexpr[2]}')
                function_list.append(START.format(id_now, split_sexpr[2]))
            else:
                agent_sexpr.append(f'{split_sexpr[2]}')
                function_list.append(START.format(id_now, split_sexpr[2]))
                pass #JOIN头实体为特殊类型date、integer实体
        agent_sexpr.append(f'(JOIN {split_sexpr[1]} {split_sexpr[2]})')
        function_list.append(JOIN.format(id_now, split_sexpr[1], 'expression'+id_now))
        return agent_sexpr, function_list  
    elif split_sexpr[0]=='AND':
        if isinstance(split_sexpr[2],list):
            temp_sexpr, temp_func = to_agent(split_sexpr[2], id_now)
            agent_sexpr += temp_sexpr
            function_list += temp_func
            split_sexpr[2] = f'{agent_sexpr[-1]}'
        else:
            agent_sexpr.append(f'{split_sexpr[2]}')
            function_list.append(START.format(id_now, split_sexpr[2]))
            pass
        if isinstance(split_sexpr[1],list): 
            id_new = '1' if id_now == '' else str(int(id_now)+1)
            split_sexpr[1],new_func = to_agent(split_sexpr[1], id_new)
            for rel in split_sexpr[1]:
                agent_sexpr.append(f'{rel} {split_sexpr[2]}')
            for func in new_func:
                function_list.append(func)
            split_sexpr[1] = split_sexpr[1][-1]
        else:
            id_new = '1' if id_now == '' else str(int(id_now)+1)
            agent_sexpr.append(f'{split_sexpr[1]} {split_sexpr[2]}')
            function_list.append(START.format(id_new, split_sexpr[1]))
            pass # AND后是type表示的实体
        agent_sexpr.append(f'(AND {split_sexpr[1]} {split_sexpr[2]})')
        function_list.append(AND.format(id_now, 'expression' + id_new, 'expression' + id_now))
        return agent_sexpr, function_list  
    elif split_sexpr[0] in ['ARGMAX', 'ARGMIN']:
        if isinstance(split_sexpr[1],list):
            temp_sexpr, temp_func = to_agent(split_sexpr[1])
            agent_sexpr += temp_sexpr
            function_list += temp_func
            split_sexpr[1] = f'{agent_sexpr[-1]}'
        else:
            agent_sexpr.append(f'{split_sexpr[1]}')
            function_list.append(START.format(id_now, split_sexpr[1]))
            pass # ARGMAX后是type表示的实体
        if split_sexpr[2][0]=="R":
            split_sexpr[2] = f'(R {split_sexpr[2][1]})'
        elif isinstance(split_sexpr[2],list):                    
            split_sexpr[2],_ = to_agent(split_sexpr[2])
            split_sexpr[2] = split_sexpr[2][-1]
            # for rel in split_sexpr[2]:
            #     agent_sexpr.append(f'({split_sexpr[0]} {split_sexpr[1]} {rel})')
            # return agent_sexpr
            pass
        else:
            pass # ARGMAX的关系正常就是一跳
        agent_sexpr.append(f'({split_sexpr[0]} {split_sexpr[1]} {split_sexpr[2]})')
        function_list.append(ARG.format(id_now, split_sexpr[0], 'expression' + id_now, split_sexpr[2]))
        return agent_sexpr, function_list
    elif split_sexpr[0] in ['le', 'lt', 'ge', 'gt']:
        if split_sexpr[1][0]=="R":
            split_sexpr[1] = f'(R {split_sexpr[1][1]})'
        elif isinstance(split_sexpr[1],list):                    
            split_sexpr[1],_ = to_agent(split_sexpr[1])
            split_sexpr[1] = split_sexpr[1][-1]
            pass
        else:
            pass 
        agent_sexpr.append(f'{split_sexpr[2]}')
        function_list.append(START.format(id_now, split_sexpr[2]))
        agent_sexpr.append(f'({split_sexpr[0]} {split_sexpr[1]} {split_sexpr[2]})')
        function_list.append(CMP.format(id_now, split_sexpr[0], split_sexpr[1], 'expression'+id_now))
        return agent_sexpr, function_list                     
    elif split_sexpr[0]=='TC':
        if isinstance(split_sexpr[1],list):
            temp_sexpr, temp_func = to_agent(split_sexpr[1])
            agent_sexpr += temp_sexpr
            function_list += temp_func
            split_sexpr[1] = f'{agent_sexpr[-1]}'
        else:
            agent_sexpr.append(f'{split_sexpr[1]}')
            function_list.append(START.format(id_now, split_sexpr[1]))
            pass
        if split_sexpr[2][0]=="R":
            split_sexpr[2] = f'(R {split_sexpr[2][1]})'
        elif isinstance(split_sexpr[2],list):                    
            split_sexpr[2],_ = to_agent(split_sexpr[2])
            split_sexpr[2] = split_sexpr[2][-1]
            pass
        else:
            pass 
        agent_sexpr.append(f'(TC {split_sexpr[1]} {split_sexpr[2]} {split_sexpr[3]})')
        function_list.append(TC.format(id_now, 'expression'+id_now, split_sexpr[2], split_sexpr[3]))
        return agent_sexpr, function_list
    elif split_sexpr[0]=='COUNT':
        if isinstance(split_sexpr[1],list):
            temp_sexpr, temp_func = to_agent(split_sexpr[1])
            agent_sexpr += temp_sexpr
            function_list += temp_func
            split_sexpr[1] = f'{agent_sexpr[-1]}'
        else:
            pass
        agent_sexpr.append(f'(COUNT {split_sexpr[1]})')
        function_list.append(COUNT.format(id_now, 'expression'+id_now))
        return agent_sexpr, function_list
    else:
        pass

def merge_all_data_for_logical_form_generation(dataset, split):
    if dataset == 'WebQSP':
        from utils.parsing.parse_sparql_webqsp import augment_with_s_expr_webqsp
        dataset_with_sexpr = augment_with_s_expr_webqsp(split,True)
    elif dataset == 'GrailQA':
        from utils.parsing.parse_sparql_grailqa import augment_with_s_expr_grailqa
        dataset_with_sexpr = augment_with_s_expr_grailqa(split,True)   
    elif dataset == 'GraphQ':
        from utils.parsing.parse_sparql_graphq import augment_with_s_expr_graphq
        dataset_with_sexpr = augment_with_s_expr_graphq(split,True)   
        
    merged_data_all = []

    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Processing {dataset}_{split}'):
        new_example = {}

        if not example['SExpr_execute_right']:
            continue
            
        if dataset == 'WebQSP':
            parses = example['Parses']
            shortest_idx = 0
            shortest_len = 9999
            for i in range(len(parses)):
                if 'SExpr_execute_right' in parses[i] and parses[i]['SExpr_execute_right']:
                    if len(parses[i]['Sparql']) < shortest_len:
                        shortest_idx = i
                        shortest_len = len(parses[i]['Sparql'])     
            qid = example['QuestionId']
            question = example['ProcessedQuestion']                   
            sexpr = parses[shortest_idx]['SExpr']
            sparql = parses[shortest_idx]['Sparql']
            answer = [x['AnswerArgument'] for x in parses[shortest_idx]['Answers']]
        elif dataset == 'GrailQA' or dataset == 'GraphQ':             
            qid = example['qid']
            question = example['question']
            sexpr = example['s_expression']
            sparql = example['sparql_query']
            answer = example["answer"]                
        
        try:              
            if dataset == 'WebQSP':
                from utils.executor.logic_form_util import lisp_to_sparql
            elif dataset=='GrailQA' or dataset=='GraphQ':
                from utils.executor.logic_form_util_grailqa import lisp_to_sparql
            sparql_query = lisp_to_sparql(sexpr)
            denotation = execute_query_with_odbc(sparql_query)
            denotation = [str(res).replace("http://rdf.freebase.com/ns/",'') for res in denotation]   
            if set(denotation)!=set(answer):
                print(sexpr)
                continue
        except:
            print(sexpr)
            continue
        sparql = sparql_query            
                    
        split_sexpr = lisp_to_nested_expression(sexpr)
        agent_sexpr, func_list = to_agent(split_sexpr)
        func_list.append('expression = STOP(expression)')
        new_sexpr = functions_to_expression(func_list, 'expression')
        
        if agent_sexpr[-1] != sexpr:
            pass
        if new_sexpr != sexpr:
            print(sexpr)
            continue
        
        new_example['ID']=qid
        new_example['question'] = question
        new_example['answer'] = answer
        new_example['sparql'] = sparql
        new_example['sexpr'] = sexpr
        new_example['function_list'] = func_list
        
        if dataset == 'GrailQA' and split == 'test':
            new_example['level'] = example["level"]
            
        if len(answer) == 0:
            continue

        merged_data_all.append(new_example)
    
    merged_data_dir = f'dataset/{dataset}/processed'
    if not os.path.exists(merged_data_dir):
        os.makedirs(merged_data_dir)
    merged_data_file = f'{merged_data_dir}/{dataset}_{split}.json'
    
    print(len(merged_data_all))
    print(f'Wrinting merged data to {merged_data_file}...')
    dump_json(merged_data_all,merged_data_file,indent=4)
    print('Writing finished')

if __name__=='__main__':
    args = _parse_args()
    merge_all_data_for_logical_form_generation(dataset=args.dataset, split="train")
    merge_all_data_for_logical_form_generation(dataset=args.dataset, split="test")
