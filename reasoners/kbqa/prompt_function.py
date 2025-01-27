import re
from utils.executor.sparql_executor import get_label_with_odbc
from utils.executor.sparql_executor import execute_query_with_odbc
from utils.database import functions_to_expression

def ent_type(ent_id):
    if ent_id.startswith('m.') or ent_id.startswith('g.'):
        return 'entity'
    elif ent_id.startswith('"') and (ent_id.endswith('"') or ent_id.endswith('"@en')):
        return 'name'
    elif '^^' in ent_id:
        return 'literal'
    elif '<http:' in ent_id:
        return 'url'
    elif '.' in ent_id:
        return 'onto'
    elif ent_id.isdigit() or (ent_id[0] == '-' and ent_id[1:].isdigit()):
        return 'int'

def prompt_function(function_list):
    scratchpad = ''
    entities = []
    for step_n, func in enumerate(function_list):
        step_n += 1
        if "START" in func:
            argument = re.findall(f"expression(.*?) = START\(\'(.*?)\'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should identify a topic entity from the question to start a new expression.\nAction{step_n}: Extract_entity '
            entity = get_label_with_odbc(argument[1]) if argument[1].startswith('m.') or argument[1].startswith('g.') else argument[1]
            action = f'[ {entity} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = START('{argument[1]}')\n" # | Start Excuted Answers: {entity} (total 1 answers)
            scratchpad += f'{plan}{action}{observation}'
            entities.append((entity,argument[1]))
        elif "JOIN(" in func:
            argument = re.findall(f"expression(.*?) = JOIN\(\'(.*?)\', expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should find the one-hop relation that is connected to the current expression.\nAction{step_n}: Find_relation '
            relation = argument[1] if '(R ' not in argument[1] else re.findall(f"\(R (.*?)\)", argument[1])[0]
            action = f'[ {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = JOIN('{argument[1]}', expression{argument[2]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "AND" in func:
            argument = re.findall(f"expression(.*?) = AND\(expression(.*?), expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should merge these two expressions.\nAction{step_n}: Merge '
            expression1, expression2 = f'expression{argument[1]}', f'expression{argument[2]}'
            action = f'[ {expression1} | {expression2} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = AND({expression1}, {expression2})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "ARG" in func:
            argument = re.findall(f"expression(.*?) = ARG\(\'(.*?)\', expression(.*?), \'(.*?)\'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a sorting operation and impose a constraint to output either the maximum or minimum value.\nAction{step_n}: Order '
            mode, relation = argument[1], argument[3]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = ARG('{mode}', expression{argument[2]}, '{relation}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'                
        elif "CMP" in func:
            argument = re.findall(f"expression(.*?) = CMP\(\'(.*?)\', \'(.*?)\', expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a numerical comparison to determine the range.\nAction{step_n}: Compare '
            mode_dict = {'le':'LESS EQUAL', 'ge':'GREATER EQUAL', 'lt':'LESS THAN', 'gt':'GREATER THAN'}
            mode, relation = mode_dict[argument[1]], argument[2]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = CMP('{argument[1]}', '{relation}', expression{argument[3]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "TC" in func:
            argument = re.findall(f"expression(.*?) = TC\(expression(.*?), \'(.*?)\', \'(.*?)\'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should add a time constraint.\nAction{step_n}: Time_constraint '
            relation, time = argument[2], argument[3]
            action = f'[ {relation} | {time} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = TC(expression{argument[1]}, '{relation}', '{time}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
            if time != 'NOW':
                entities.append((time,argument[3]))
        elif "COUNT" in func:
            argument = re.findall(f"expression(.*?) = COUNT\(expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a counting operation to determine the number of answers.\nAction{step_n}: Count '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = COUNT({expression})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "STOP" in func:
            argument = re.findall(f"expression(.*?) = STOP\(expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we conclude that it is appropriate to end and output the expression.\nAction{step_n}: Finish '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = STOP({expression})\n" # | Final Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        else:
            pass
    # print(scratchpad)
    return scratchpad, entities

def get_next_r_relations(func_list,dataset):
    id_now = func_list[-1].split(" = ")[0].replace('expression','')
    func_list = [] + func_list
    func_list.append(f'expression{id_now} = STOP(expression{id_now})')
    sexpr = functions_to_expression(func_list, f'expression{id_now}')     
    try:    
        if dataset == 'WebQSP' or dataset == 'GraphQ' or dataset == 'GrailQA':
            from utils.executor.logic_form_util import lisp_to_sparql
        if not sexpr.startswith('('):
            if ent_type(sexpr) in ['entity','onto']:
                sparql_query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\nns:" + sexpr + " ?relation ?y .\n}"
            elif ent_type(sexpr) in ['int','url']:
                func_list = func_list[:-1] + [f"expression{id_now} = JOIN('(R RELATION)', expression{id_now})"] + [func_list[-1]]
                sexpr = functions_to_expression(func_list, f'expression{id_now}')  
                sparql_query = lisp_to_sparql(sexpr).replace('ns:RELATION','?relation').replace('SELECT DISTINCT ?x','SELECT DISTINCT ?relation')
        else:
            original_sparql = lisp_to_sparql(sexpr).replace('PREFIX ns: <http://rdf.freebase.com/ns/>\n','')
            sparql_query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n{\n" + original_sparql + "\n}\n\n?x ?relation ?y .\n}"
        denotation = execute_query_with_odbc(sparql_query)
        denotation = [str(res).replace("http://rdf.freebase.com/ns/",'') for res in denotation if "http://rdf.freebase.com/ns/" in str(res)]
    except:
        print('Error in sparql query')
        denotation = []
    return denotation

def get_next_relations(func_list,dataset):
    id_now = func_list[-1].split(" = ")[0].replace('expression','')
    func_list = [] + func_list
    func_list.append(f'expression{id_now} = STOP(expression{id_now})')
    sexpr = functions_to_expression(func_list, f'expression{id_now}')     
    try:    
        if dataset == 'WebQSP' or dataset == 'GraphQ' or dataset == 'GrailQA':
            from utils.executor.logic_form_util import lisp_to_sparql
        if not sexpr.startswith('('):
            if ent_type(sexpr) in ['entity','onto']:
                sparql_query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n?y ?relation ns:" + sexpr + " .\n}"
            elif ent_type(sexpr) in ['int','url']:
                func_list = func_list[:-1] + [f"expression{id_now} = JOIN('RELATION', expression{id_now})"] + [func_list[-1]]
                sexpr = functions_to_expression(func_list, f'expression{id_now}')  
                sparql_query = lisp_to_sparql(sexpr).replace('ns:RELATION','?relation').replace('SELECT DISTINCT ?x','SELECT DISTINCT ?relation')
        else:
            original_sparql = lisp_to_sparql(sexpr).replace('PREFIX ns: <http://rdf.freebase.com/ns/>\n','')
            sparql_query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n{\n" + original_sparql + "\n}\n\n?y ?relation ?x .\n}"
        denotation = execute_query_with_odbc(sparql_query)
        denotation = [str(res).replace("http://rdf.freebase.com/ns/",'') for res in denotation if "http://rdf.freebase.com/ns/" in str(res)]
    except:
        print('Error in sparql query')
        denotation = []
    return denotation

def get_sexpr(prompt,entity):
    if prompt == '':
        return '@BAD_EXPRESSION'
    func_list = []
    for step in prompt.split('\n'):
        if 'Observation' in step:
            func_list.append(step.split(': ')[1])
    id_now = func_list[-1].split(" = ")[0].replace('expression','')
    sexpr = functions_to_expression(func_list, f'expression{id_now}')
    
    for ent,ent_id in entity:
        if ent_id in sexpr and ent:
            sexpr = sexpr.replace(f' {ent_id} ',f' <{ent.replace(" ","_")}> ')
            sexpr = sexpr.replace(f' {ent_id})',f' <{ent.replace(" ","_")}>)')
    return sexpr

def execute_function_list(func_list,dataset):
    id_now = func_list[-1].split(" = ")[0].replace('expression','')
    sexpr = functions_to_expression(func_list, f'expression{id_now}')
    try:    
        if dataset == 'WebQSP' or dataset == 'GraphQ':
            from utils.executor.logic_form_util import lisp_to_sparql
        elif dataset == 'GrailQA':
            from utils.executor.logic_form_util_grailqa import lisp_to_sparql
        sparql_query = lisp_to_sparql(sexpr)
        denotation = execute_query_with_odbc(sparql_query)
        denotation = [str(res).replace("http://rdf.freebase.com/ns/",'') for res in denotation]
    except:
        print('Error in sparql query')
        denotation = []    
    return {"sexpr": sexpr, "answers": denotation if denotation!=["None"] else []}

def test_execute_function_list(func_list,dataset):
    id_now = func_list[-1].split(" = ")[0].replace('expression','')
    func_list = [] + func_list
    func_list.append(f'expression{id_now} = STOP(expression{id_now})')
    sexpr = functions_to_expression(func_list, f'expression{id_now}')
    try:    
        if dataset == 'WebQSP' or dataset == 'GraphQ':
            from utils.executor.logic_form_util import lisp_to_sparql
        elif dataset == 'GrailQA':
            from utils.executor.logic_form_util_grailqa import lisp_to_sparql
        sparql_query = lisp_to_sparql(sexpr)
        denotation = execute_query_with_odbc(sparql_query)
        denotation = [str(res).replace("http://rdf.freebase.com/ns/",'') for res in denotation] #  if "http://rdf.freebase.com/ns/" in str(res)
    except:
        print('Error in sparql query')
        denotation = []    
    return denotation

def function_list2sexpr(func_list,dataset):
    id_now = func_list[-1].split(" = ")[0].replace('expression','')
    func_list = [] + func_list
    func_list.append(f'expression{id_now} = STOP(expression{id_now})')
    sexpr = functions_to_expression(func_list, f'expression{id_now}')   
    return sexpr



