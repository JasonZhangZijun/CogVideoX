import torch
import gradio as gr
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from PIL import Image
import os

# 设置模型路径和设备
model_path = "/home/zijun/workspace/CogVideo/checkpoints/CogVideoX-5B-I2V"
device = "cuda:5" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(0)

# 加载模型组件
transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16).to(device)
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16).to(device)

# 初始化 I2V pipeline
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_path,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
).to(device)

# 启用优化
pipe.enable_sequential_cpu_offload()

# 设定输出目录
os.makedirs("./output", exist_ok=True)

def infer(prompt: str, image: Image, num_inference_steps: int = 50):
    """ 执行 I2V 推理 """
    # 处理输入图片
    image = image.convert("RGB")  # 确保是 RGB 格式
    image.save("./output/input_image.jpg")  # 临时保存图片
    image = load_image("./output/input_image.jpg")

    # 生成视频
    video = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=6.0,
        use_dynamic_cfg=True,
        num_inference_steps=num_inference_steps
    ).frames[0]

    # 导出视频
    video_path = "./output/generated_video.mp4"
    export_to_video(video, video_path, fps=8)

    return video_path

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 CogVideoX-5B-I2V Gradio Demo")

    with gr.Row():
        image_input = gr.Image(label="输入图片", type="pil")
        prompt = gr.Textbox(label="输入文本描述", placeholder="输入你的描述...", lines=3)

    num_steps = gr.Slider(10, 100, value=50, step=10, label="推理步数")
    generate_button = gr.Button("🎬 生成视频")
    video_output = gr.Video(label="生成的视频")

    generate_button.click(
        infer,
        inputs=[prompt, image_input, num_steps],
        outputs=[video_output],
    )

if __name__ == "__main__":
    demo.launch()
