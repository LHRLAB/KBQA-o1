from utils.prompt import exec_globals

BAD_EXPRESSION = '@BAD_EXPRESSION'
BAD_SPARQL = '@BAD_SPARQL'


def functions_to_expression(query_functions, query_target):
    try:
        local = {}
        exec('\n'.join(query_functions), exec_globals, local)
        return local[query_target]
    except:
        return BAD_EXPRESSION
