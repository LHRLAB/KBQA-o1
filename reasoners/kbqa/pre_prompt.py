from langchain.prompts import PromptTemplate

INSTRUCTION_KBQA = """I want you to be a good knowledge graph query generator, solving a knowledge base question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be eight types : 
(1) Extract_entity [ entity name ], where we identify a topic entity from the question to start a new expression.
(2) Find_relation [ relation name ], where we find the one-hop relation that is connected to the current expression.
(3) Merge [ expression1 | expression2 ], where we merge these two expressions.
(4) Order [ mode | relation ], where we perform a sorting operation and impose a constraint to output either the maximum or minimum value.
(5) Compare [ mode | relation ], where we perform a numerical comparison to determine the range.
(6) Time_constraint [ relation | time ], where we add a time constraint.
(7) Count [ expression ], where we perform a counting operation to determine the number of answers.
(8) Finish [ expression ], where we conclude that it is appropriate to end and output the expression.
Here are some examples:
{examples}
(END OF EXAMPLES)
<input>
Question: {question}
<output>
{scratchpad}"""

agent_prompt_kbqa = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = INSTRUCTION_KBQA,
                        )

EXAMPLE_INSTRUCTION_KBQA ="""<input>
Question: {question}
<output>
{scratchpad}"""

example_prompt_kbqa = PromptTemplate(
                        input_variables=["question", "scratchpad"],
                        template = EXAMPLE_INSTRUCTION_KBQA,
                        )

PLAN_LIST = {
    "Extract_entity": "Thought{}: At this step, we should identify a topic entity from the question to start a new expression.\nAction{}: Extract_entity ", 
    "Find_relation": "Thought{}: At this step, we should find the one-hop relation that is connected to the current expression.\nAction{}: Find_relation ", 
    "Merge": "Thought{}: At this step, we should merge these two expressions.\nAction{}: Merge ", 
    "Order": "Thought{}: At this step, we should perform a sorting operation and impose a constraint to output either the maximum or minimum value.\nAction{}: Order ", 
    "Compare": "Thought{}: At this step, we should perform a numerical comparison to determine the range.\nAction{}: Compare ", 
    "Time_constraint": "Thought{}: At this step, we should add a time constraint.\nAction{}: Time_constraint ", 
    "Count": "Thought{}: At this step, we should perform a counting operation to determine the number of answers.\nAction{}: Count ", 
    "Finish": "Thought{}: At this step, we conclude that it is appropriate to end and output the expression.\nAction{}: Finish "
}