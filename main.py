from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import transformers
import torch

app = FastAPI()

model_path = "/mnt/data/gaomeng/models/Meta-Llama-3.1-8B-Instruct"
try:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # 输出模型的原始精度
    original_dtype = next(model.parameters()).dtype
    print(f"模型和分词器加载成功。模型的原始精度为: {original_dtype}")
except Exception as e:
    print(f"加载模型或分词器时出错: {e}")

# 添加 pad_token，如果没有的话
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    print(f"成功添加 pad_token，pad_token_id 为：{tokenizer.pad_token_id}")

# 尝试将模型转换为 float16
to_float16 = True
if to_float16:
    try:
        model.half()  # 使用半精度浮点数
        print("模型成功转换为 float16。")
    except Exception as e:
        print(f"模型转换为 float16 时出错: {e}")

# 尝试将模型转换为 int8（可选）
to_int8 = False
if to_int8:
    try:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print("模型成功转换为 int8。")
    except Exception as e:
        print(f"模型转换为 int8 时出错: {e}")

# 检测是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备为: {device}")

try:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # 如果有 GPU 则使用 GPU，否则使用 CPU
    )
    print("生成pipeline成功。")
except Exception as e:
    print(f"生成pipeline时出错: {e}")

class ChatMessage(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: list[ChatMessage]

def extract_system_reply(text, question):
    print(f"Generated Text: {text}")
    print(f"User Question: {question}")
    start_index = text.find(f"user: {question}")

    if start_index != -1:
        start_index = text.find("system:", start_index)
        if start_index != -1:
            end_index = text.find("user:", start_index)
            if end_index == -1:
                end_index = len(text)
            system_response = text[start_index + 7 : end_index].strip()
            return system_response
    return "No matching reply found."
  
def processing_message(request_body):
    messages = [msg.dict() for msg in request_body.messages[-7:]]
    m2_refine = [{'role': 'system', 'content': '我是一个严谨认真耐心有礼貌的博士,会帮助你解决所有问题!'},
      {'role': 'user', 'content': '你能回答我的问题吗?'},
      {'role': 'system', 'content': '当然!我会简洁准确的回答你的问题'},
    ]
    all_messages = [msg.dict() for msg in request_body.messages[-7:]]
    for msg in all_messages:
        if msg['role'] == 'user' and m2_refine[-1]['role'] == 'system':
            m2_refine.append({'role': 'user', 'content': msg['content']})
        elif msg['role'] != 'user' and m2_refine[-1]['role'] == 'user':
            m2_refine.append({'role': 'system', 'content': msg['content']})
    formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in m2_refine])
    return formatted_messages

@app.post("/generate")
async def generate_text(request_body: RequestBody):
    question = request_body.messages[-1].dict()['content']  # 修改为 -1 以获取最新消息
    formatted_messages = processing_message(request_body)
    try:
        outputs = pipeline(
            formatted_messages,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
        generated_text = outputs[0]["generated_text"]
        print("生成文本成功。")
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return {"generated_text": "Sorry, there was an error processing your request."}

    responce = extract_system_reply(generated_text, question)
    return {"generated_text": responce}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("frontend.html", "r") as file:
            html_content = file.read()
        print("HTML文件加载成功。")
    except Exception as e:
        print(f"加载HTML文件时出错: {e}")
        return HTMLResponse(content="Error loading HTML file", status_code=500)
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
        print("FastAPI 服务启动成功。")
    except Exception as e:
        print(f"启动FastAPI服务时出错: {e}")
