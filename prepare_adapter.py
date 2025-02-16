from peft import LoraConfig
import json
import os

# 定义 LoRA 配置
transformer_lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    init_lora_weights=True,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"]
)

# **确保 `config_path` 是字符串**
config_dir = "/home/zijun/workspace/CogVideo/finetune_output/checkpoint-9/"
config_path = os.path.join(config_dir, "adapter_config.json")

# **确保目录存在**
os.makedirs(config_dir, exist_ok=True)

# **保存 JSON**
with open(config_path, "w") as f:
    json.dump(transformer_lora_config.to_dict(), f)

print("LoRA adapter_config.json saved successfully at:", config_path)
