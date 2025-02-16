import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
model_path = "THUDM/CogVideoX-5b-I2V"
torch.cuda.set_device(2)
device = "cuda:2"

transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16).to(device)
text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16).to(device)

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_path,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
).to(device)
pipe.enable_sequential_cpu_offload()
prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg")
video = pipe(image=image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

export_to_video(video, "output.mp4", fps=8)