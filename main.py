import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import replicate
from dotenv import load_dotenv
import base64

# 加载环境变量
load_dotenv()

app = FastAPI()



# 发型提示词库 (Prompt Engineering)
# 这里对应之前搜索到的流行发型
HAIRSTYLE_PROMPTS = {
    "women_one_length": "a woman with short one-length bob haircut, sharp edges, photorealistic, 8k, high quality",
    "women_french_perm": "a woman with french perm, wool curly hair, vintage vibes, photorealistic, 8k",
    "women_layered": "a woman with trendy layered clavicle haircut, airy texture, photorealistic, 8k",
    "men_texture_crop": "a man with texture crop haircut, korean style, clean fade, photorealistic, 8k",
    "men_wolf_cut": "a man with edgy wolf cut mullet hairstyle, punk style, photorealistic, 8k",
}

def run_instant_id(image_url, prompt):
    """调用 Replicate 上的 InstantID 模型"""
    try:
        # 使用 wangfuyun/instantid 模型，这是目前保持人脸一致性最好的模型
        output = replicate.run(
            "wangfuyun/instantid:c6b5d2b7459910fec94432e9e1203c3cdce92d6db20f714f1355747990b52fa6",
            input={
                "image": image_url,
                "prompt": prompt,
                "negative_prompt": "bad quality, distorted face, low resolution, blurry, ugly, extra ears",
                "style_strength": 20, # 风格强度
                "ip_adapter_scale": 0.8, # 保持人脸相似度的权重 (0-1)
                "num_inference_steps": 30,
                "controlnet_selection": "depth" # 使用深度图控制结构
            }
        )
        return output[0] # 返回生成的图片 URL
    except Exception as e:
        print(f"AI 生成出错: {e}")
        return None

@app.post("/generate")
async def generate_hairstyle(
    file: UploadFile = File(...), 
    hairstyle: str = Form(...)
):
    # 1. 验证发型是否存在
    if hairstyle not in HAIRSTYLE_PROMPTS:
        return JSONResponse(content={"error": "未知的发型选择"}, status_code=400)
    
    prompt = HAIRSTYLE_PROMPTS[hairstyle]
    
    # 2. 读取图片并转换为 Base64 (Replicate 可以直接接收 URL 或 Base64 URI)
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")
    image_uri = f"data:image/jpeg;base64,{base64_image}"

    print(f"正在生成发型: {hairstyle} ...")
    
    # 3. 调用 AI
    result_url = run_instant_id(image_uri, prompt)
    
    if result_url:
        return {"status": "success", "image_url": result_url}
    else:
        return JSONResponse(content={"error": "生成失败，请检查后端日志"}, status_code=500)

from fastapi.responses import FileResponse

@app.get("/")
async def read_index():
    return FileResponse('index.html')

if __name__ == "__main__":
    print("服务器启动中... 请访问 http://127.0.0.1:8000/static/index.html")
    uvicorn.run(app, host="0.0.0.0", port=8000)
