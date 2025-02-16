import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
import os
from peft import PeftModel


model_path = "/home/zijun/workspace/CogVideo/checkpoints/CogVideoX-5B-I2V"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(2)

lora_model_path = "finetune_output/checkpoint-9"
transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16).to(device)
transformer = PeftModel.from_pretrained(transformer, lora_model_path, local_files_only=True, adapter_weights_name="pytorch_lora_weights.safetensors").to(device)
# transformer = PeftModel.from_pretrained(transformer, lora_model_path, local_files_only=True, adapter_weights_name="adapter_model.safetensors").to(device)
transformer = transformer.merge_and_unload()
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16).to(device)

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_path,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
).to(device)
# pipe.enable_sequential_cpu_offload()
prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image("/home/zijun/workspace/CogVideo/inference/astronaut.jpg")
video = pipe(image=image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]
export_to_video(video, "finetuned.mp4", fps=8)

