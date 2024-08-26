from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import transformers
import torch

app = FastAPI()

# 使用本地模型路径
model_path = "/mnt/data/gaomeng/models/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,  # 如果模型路径包含tokenizer
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

class ChatMessage(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: list[ChatMessage]

def extract_system_reply(text, question):
    # 找到用户问题的位置
    start_index = text.find(f"user: {question}")
    
    if start_index != -1:
        # 找到从问题之后的系统回复的位置
        start_index = text.find("system:", start_index)
        if start_index != -1:
            # 找到下一个“user:”或文本结束的位置
            end_index = text.find("user:", start_index)
            if end_index == -1:
                end_index = len(text)
            
            # 提取出系统的回复
            system_response = text[start_index + 7 :end_index].strip()
            return system_response
    
    return "No matching reply found."
  
def processing_message(request_body):
    messages = [msg.dict() for msg in request_body.messages[-7:-1]]  # 将消息对象转换为字典
    m2_refine = [{'role': 'system', 'content': 'You are a pirate who always responds in pirate speak!'},
      {'role': 'user', 'content': 'Who are you?'},
      {'role': 'system', 'content': 'I am the pirate! Come sail with me!!'},
      {'role': 'user', 'content': 'Can you help me?'},
      {'role': 'system', 'content': 'Of course, my dear crew, let the great captain assist you!'},
    ]
    all_messages = [msg.dict() for msg in request_body.messages[-7:-1]]
    for msg in all_messages:
      if msg['role'] == 'user' and m2_refine[-1]['role'] == 'system':
        m2_refine.append({'role': 'user', 'content': msg['content']})
      elif msg['role'] != 'user' and m2_refine[-1]['role'] == 'user':
        m2_refine.append({'role': 'system', 'content': msg['content']})
    # 将消息列表转换为单个字符串输入
    formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in m2_refine])
    return formatted_messages
    


@app.post("/generate")
async def generate_text(request_body: RequestBody):
    question = request_body.messages[-1].dict()['content']
    formatted_messages = processing_message(request_body)
    
    # 生成文本
    outputs = pipeline(
        formatted_messages,
        max_new_tokens=128,  # 限制生成长度，避免过多重复
        do_sample=True,      # 允许采样，避免模型总是输出相同的内容
        temperature=0.7      # 控制采样的随机性，值越高越随机
    )
    
    # 提取生成的文本
    generated_text = outputs[0]["generated_text"]

    responce = extract_system_reply(generated_text, question)
    return {"generated_text": responce}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
