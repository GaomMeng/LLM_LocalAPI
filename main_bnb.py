from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import transformers
import torch
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

app = FastAPI()

def load_model(model_path: str, precision: str):
    try:
        if precision == "8":
            # 使用 BitsAndBytesConfig 配置 8-bit 量化
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # 控制量化阈值，默认为6.0，可以根据需求调整
                llm_int8_skip_modules=["lm_head"],  # 避免量化特定模块
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=quantization_config, device_map="auto"
            )
        elif precision == "16":
            # 加载 16-bit 精度模型
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        elif precision == "32":
            # 加载 32-bit 精度模型
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, device_map="auto"
            )
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        
        # 输出模型的原始精度
        original_dtype = next(model.parameters()).dtype
        print(f"模型和分词器加载成功。模型的原始精度为: {original_dtype}")
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型或分词器时出错: {e}")
        raise

model_path = "/mnt/data/gaomeng/models/Meta-Llama-3.1-8B-Instruct"
precision = "8"  # 默认使用 8-bit 量化
model, tokenizer = load_model(model_path, precision)

# 添加 pad_token，如果没有的话
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    print(f"成功添加 pad_token，pad_token_id 为：{tokenizer.pad_token_id}")

# 检测是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备为: {device}")

# 创建 pipeline 时不再指定 device_map 参数
try:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
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
      {'role': 'user', 'content': '你能帮助我吗?'},
      {'role': 'system', 'content': '当然!我乐于助人且很有耐心彬彬有礼.'},
    ]
    all_messages = [msg.dict() for msg in request_body.messages[-7:-1]]
    for msg in all_messages:
        if msg['role'] == 'user' and m2_refine[-1]['role'] == 'system':
            m2_refine.append({'role': 'user', 'content': msg['content']})
        elif msg['role'] != 'user' and m2_refine[-1]['role'] == 'user':
            m2_refine.append({'role': 'system', 'content': msg['content']})
    formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in m2_refine])
    return formatted_messages

@app.post("/generate")
async def generate_text(request_body: RequestBody):
    question = request_body.messages[-1].dict()['content']
    formatted_messages = processing_message(request_body)
    try:
        outputs = pipeline(
            formatted_messages,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )
        generated_text = outputs[0]["generated_text"]
        print("生成文本成功。")
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return {"error": "生成文本时出错"}

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
    import sys

    # 从命令行参数获取精度设置
    if len(sys.argv) > 1:
        precision = sys.argv[1]
        print(f"选择的精度为: {precision}")
        model, tokenizer = load_model(model_path, precision)
    else:
        print("未指定精度参数，使用默认精度 8-bit")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
        print("FastAPI 服务启动成功。")
    except Exception as e:
        print(f"启动FastAPI服务时出错: {e}")
