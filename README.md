# CogVideoX with Causal Attention Mask
**Name:** Zhang Zijun   **Student ID:** A69036367   **Date:** February 15, 2025  

This repository is a modified version of **CogVideoX** designed to test causal attention mechanisms in video generation models. 

This README only provides **instructions on running the modified CogVideoX version**, while a separate document explains **modifications to the attention mechanism and challenges encountered during development**.

---

## Installation and Setup

Ensure your environment meets the required dependencies:

```bash
pip install -r requirements.txt
```

#### Modifying `diffusers`

The original CogVideoX model relies on the `diffusers` library. However, since we need to modify certain functions, we use a **locally installed** version:

```bash
pip uninstall diffusers
cd diffusers
pip install -e .
```

#### Download Pretrained Models

For **Image-to-Video (I2V)** generation, use one of the following models:

- `CogVideoX-5B-I2V`
- `CogVideoX1.5-5B-I2V`

(Available from: `THUDM/CogVideoX-5B-I2V` and`THUDM/CogVideoX1.5-5B-I2V` and  on Hugging Face)

------

## Quick Test for I2V Model

To run a simple inference test using the **I2V model**, execute:

```bash
python i2v.py
```

------

## Running the Gradio Interface

To launch the **Gradio-based web demo**, execute:

```bash
python gradio_demo.py
```

This will start a web service at:

```
http://localhost:7860/
```

------

## Fine-tuning the I2V Model

### 1. Prepare Your Dataset

Depending on whether you're working with **Text-to-Video (T2V)** or **Image-to-Video (I2V)** tasks, the dataset structure will differ. Below is the standard format:

```
.
├── prompts.txt
├── videos
├── videos.txt
├── images     # (Optional) For I2V: If not provided, the first frame of the video will be extracted.
└── images.txt # (Optional) For I2V: If not provided, the first frame of the video will be extracted.
```

#### Example Dataset: **Disney Steamboat Willie**

For demonstration, I used the [Disney Steamboat Willie dataset](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset), which follows this structure:

```
.
├── prompts.txt
├── videos
├── videos.txt
```

Since this dataset **lacks images and `images.txt`**, you can handle this in one of two ways:

1. Extract images from videos

    using:

   ```bash
   python finetune/scripts/extract_images.py
   ```

2. Modify training parameters:

   - Comment out `--image_column "images.txt"` in the training script to **default to the first video frame** as input.

------

### 2. Start Fine-tuning

Modify `train_ddp_i2v.sh` to include the appropriate **model path, dataset path, and training parameters**.

Then, start training with:

```bash
bash train_ddp_i2v.sh
```

------

## Testing Your Finetuned I2V Model

Since we use **LoRA fine-tuning**, the trained checkpoint includes **LoRA weights**. To use the fine-tuned model, we need to properly configure LoRA adapters.

### 1. Generate the `adapter_config.json`

After fine-tuning, the checkpoint **does not automatically include `adapter_config.json`**, which is required for defining the LoRA configuration.

To generate this file, run:

```bash
python prepare_adapter.py
```

Internally, this script creates an adapter config based on:

```python
transformer_lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    init_lora_weights=True,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"]
)
```

------

### 2. Run Inference with the Fine-tuned Model

Use the `cli_demo.py` script in the `inference/` folder to **generate a video from an image prompt**:

```bash
python inference/cli_demo.py \
    --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realized in the background. High quality, ultra-realistic detail and breathtaking movie-like camera shot." \
    --model_path "/home/zijun/workspace/CogVideo/checkpoints/CogVideoX-5B-I2V" \
    --lora_path "/home/zijun/workspace/CogVideo/finetune_output/checkpoint-9" \
    --generate_type "i2v" \
    --image_or_video_path "/home/zijun/workspace/CogVideo/inference/astronaut.jpg" \
    --output_path "./output_i2v_finetune.mp4" \
    --num_inference_steps 50 \
    --num_frames 81 \
    --guidance_scale 6.0 \
    --fps 16 \
    --dtype "float16"
```

------

