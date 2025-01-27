from llamafactory.hparams import get_infer_args
from llamafactory.chat.hf_engine import HuggingfaceEngine
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import Sequence
from llamafactory.chat import ChatModel

app = FastAPI()

chat_model = ChatModel()
device = chat_model.engine.model.device
max_batch_size = 1

def score(engine: HuggingfaceEngine, input: str, output: Sequence[str]):
    input = input = "user\n\n"+input+"assistant\n\n"
    
    prefix = input
    contents = [input + out for out in output]
    batches = [contents[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]
    acc_probs_list = []
    for contents in batches:
        bsz = len(contents)
        assert bsz <= max_batch_size, (bsz, max_batch_size)
        prompts_tokens = engine.tokenizer(contents, return_tensors='pt',add_special_tokens=False, padding=True).to(device)
        prefix_tokens = engine.tokenizer(prefix, return_tensors='pt',add_special_tokens=False, padding=True).input_ids[0].to(device)
        
        tokens = prompts_tokens
        logits = engine.model(**tokens, return_dict=True).logits
        tokens = prompts_tokens.input_ids
        acc_probs = torch.zeros(bsz).to(device)
        seq_lengths = torch.zeros(bsz).to(device)  # 用于记录每个序列的有效长度
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i-1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != engine.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
                    seq_lengths[j] += 1  # 增加有效 token 数量
        # 计算平均对数概率
        acc_probs /= (seq_lengths + 1e-8)  # 避免除以零
        acc_probs_list += acc_probs.cpu().numpy().tolist()
    acc_probs_list = [100.0+100*acc for acc in acc_probs_list]
    return acc_probs_list 

def beam(engine: HuggingfaceEngine, input: str, beam_size: int):
    messages = []
    messages.append({"role": "user", "content": input})  
    gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
        engine.model, engine.tokenizer, engine.processor, engine.template, engine.generating_args, messages, None, None, None, 
        {
         }
    )
    generate_output = engine.model.generate(
        **gen_kwargs,
        num_beams = beam_size,
        num_return_sequences = beam_size,
        return_dict_in_generate=True,
        output_scores=True,
        )
    
    response_ids = generate_output.sequences[:, prompt_length:]
    response = engine.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    scores = generate_output.sequences_scores.cpu().tolist()

    # 打印生成结果及对应得分
    for i, (text, score) in enumerate(zip(response, scores)):
        print(f"Generated Text {i+1} (Score: {score:.4f}):\n{text}\n")
        
    return zip(response, scores)

class LLMRequest(BaseModel):
    input: str
    output: Sequence[str]

@app.post(f"/{os.environ.get('MODEL_NAME', 'llm')}")
async def llm(request:LLMRequest):
    if len(request.output) == 0:
        messages = []
        messages.append({"role": "user", "content": request.input})
        response = ""
        for new_text in chat_model.stream_chat(messages):
            response += new_text
    else:
        if request.output[0] == "BEAM":
            response = beam(chat_model.engine, request.input, int(request.output[1]))
        else:
            response = score(chat_model.engine, request.input, request.output)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=int(os.environ.get("API_PORT", "8000")))
