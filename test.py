# from huggingface_hub import snapshot_download

# # 下载模型
# model_path = "/home/zijun/workspace/CogVideo/checkpoints/CCogVideoX1.5-5B-SAT"  # 你希望存放模型的目录
# snapshot_download(repo_id="THUDM/CogVideoX1.5-5B-SAT", local_dir=model_path)

# import diffusers
# import os

# print("Diffusers module path:", diffusers.__file__)
# print("Diffusers installed at:", os.path.dirname(diffusers.__file__))

# import torch
# import torch.nn.functional as F

# # 1. 读取保存的 Attention 变量
# saved_tensors = torch.load("saved_attention_tensors.pth")

# query = saved_tensors["query"]
# key = saved_tensors["key"]
# value = saved_tensors["value"]
# attention_mask = saved_tensors["attention_mask"]


# seq_len = query.shape[-2]  # 获取序列长度
# causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device) * -1e9, diagonal=1)
# causal_mask = causal_mask[None, None, :, :]  # 扩展维度以匹配 batch 和 heads

# print("Query shape:", query.shape) # torch.Size([2, 48, 17776, 64])
# print("Key shape:", key.shape) # torch.Size([2, 48, 17776, 64])
# print("Value shape:", value.shape) # torch.Size([2, 48, 17776, 64])
# # print("Attention Mask shape:", attention_mask.shape if attention_mask is not None else "None")

# 2. 重新计算 scaled dot-product attention
# hidden_state = F.scaled_dot_product_attention(
#     query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
# )

# hidden_state_mask1 = F.scaled_dot_product_attention(
#     query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True  # 这里不使用 is_causal
# )
# #hidden_state: torch.Size([2, 48, 17776, 64])
# difference = torch.abs(hidden_state_mask1 - hidden_state).sum().item()
# print("Difference between is_causal=True and manual causal_mask:", difference)
# 3. 打印 attention 权重，检查是否是 Causal Mask
# print("Attention Weights Shape:", hidden_state.shape)  # 应该是 (batch, num_heads, seq_len, seq_len)

# # print("Attention Weights Matrix (first head):")
# # print(hidden_state[0, 0, :20, :20])  # 只打印前10行前10列，确保可读


import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
import os
from peft import PeftModel


model_path = "THUDM/CogVideoX-5B-I2V"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(2)

lora_model_path = "/home/zijun/workspace/CogVideo/finetune_output/checkpoint-9/"
transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16).to(device)
transformer = PeftModel.from_pretrained(
    transformer, 
    lora_model_path, 
    local_files_only=True, 
    adapter_weights_name="pytorch_lora_weights.safetensors"
).to(device)


