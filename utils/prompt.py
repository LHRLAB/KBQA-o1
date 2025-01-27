"""
Please think and generate reasoning processes in the annotation, then use the functions defined below to generate the expression corresponding to the question step by step.
"""


def START(entity: str):
    return entity


def JOIN(relation: str, expression: str):
    return '(JOIN {} {})'.format(relation, expression)


def AND(expression: str, sub_expression: str):
    return '(AND {} {})'.format(expression, sub_expression)


def ARG(operator: str, expression: str, relation: str):
    assert operator in ['ARGMAX', 'ARGMIN']
    return '({} {} {})'.format(operator, expression, relation)


def CMP(operator: str, relation: str, expression: str):
    # assert operator in ['<', '<=', '>', '>=']
    return '({} {} {})'.format(operator, relation, expression)

def TC(expression: str, relation: str, entity: str):
    return '(TC {} {} {})'.format(expression, relation, entity)

def COUNT(expression: str):
    return '(COUNT {})'.format(expression)


def STOP(expression: str):
    return expression


# -------------------------------------------------------------

exec_globals = {
    'START': START,
    'JOIN': JOIN,
    'AND': AND,
    'ARG': ARG,
    'CMP': CMP,
    'TC': TC,
    'COUNT': COUNT,
    'STOP': STOP,
}

if __name__ == "__main__":
    expression = START('creators of lab')
    expression = JOIN('computer.web_browser.developers', expression)
    expression1 = START('lab')
    expression1 = JOIN('computer.web_browser.developed_by', expression1)
    expression = AND(expression, expression1)
    expression = JOIN('computer.web_browser.name', expression)
    expression = AND('computer.web_browser', expression)
    expression1 = START('computer.web_browser')
    expression1 = ARG('ARGMAX', expression1, 'computer.web_browser.release_date')
    expression = AND(expression, expression1)
    expression = STOP(expression)

    print(expression)
