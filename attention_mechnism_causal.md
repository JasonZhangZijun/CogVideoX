# CogVideoX Causal Attention Mechanism Modification

**Name:** Zhang Zijun   **Student ID:** A69036367   **Date:** February 15, 2025  

This project is based on **CogVideoX**, aiming to modify its **attention mechanism** from **bidirectional** to **causal** to better suit streaming video generation tasks. This document provides a detailed explanation of the modifications implemented, along with encountered issues and their solutions.

---

## 1. Attention Mechanism Modification

### 1.1 Locating the Key Code

After analyzing the structure of CogVideoX, we identified the **key files** related to **attention computation**:

- **Transformer Model**:  
  `CogVideo/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`

- **Attention Processor**:  
  `CogVideo/diffusers/src/diffusers/models/attention_processor.py`

- **Attention Processor Class Used**:  
  `CogVideoXAttnProcessor2_0`

By iterating over `transformer.named_modules()` and printing all registered modules, we confirmed that **only `CogVideoXAttnProcessor2_0` is used as the attention processor**, while other classes in `attention_processor.py` are not utilized.

---

### 1.2 Key Modification: Applying Causal Attention Mask

In the `CogVideoXAttnProcessor2_0` class, attention computation utilizes **PyTorch's built-in `scaled_dot_product_attention` function**:

```python
hidden_states = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
)
```

By default, `is_causal=False`, meaning the model applies **bidirectional attention**, which is suitable for standard Transformer architectures. However, for **streaming video generation tasks**, we require **causal attention**, ensuring that each frame **only depends on past information**, preventing the model from accessing future frames during inference.

#### **Modification Approach**

Setting `is_causal=True` enables the **upper triangular mask**, enforcing causal attention:

```python
hidden_states = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
)
```

This modification ensures that:

- Each token **can only attend to itself and previous tokens**, preventing access to future tokens.
- The **video generation process becomes streaming-friendly**, making it suitable for **online inference**.

------

### 1.3 Attention Mask Verification

By setting breakpoints in `attention_processor.py` at the attention computation step, we observed:

```python
hidden_states = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
)
```

The `query`, `key`, and `value` tensors have dimensions `(batch_size, attn.heads, sequence_length, head_dim)`, which match:

```python
torch.Size([2, 48, 17776, 64])
```

However, **the attention mask was `None`**, meaning it did not explicitly mask future tokens. Since `F.scaled_dot_product_attention` is implemented in C++, it is difficult to directly inspect the attention weights to verify if it applies an upper triangular matrix.

#### **Finding from PyTorch Documentation:**

After reviewing the documentation, I found that the equivalent **Python implementation** of `scaled_dot_product_attention` contains the following logic:

```python
if is_causal:
  assert attn_mask is None
  temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
  attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
  attn_bias.to(query.dtype)
```

This means that **when `is_causal=True`, PyTorch automatically constructs the required mask for causal attention**.

#### **Verification: Controlling Variables to Check `is_causal` Effect**

To confirm whether setting `is_causal=True` successfully modifies attention behavior, I conducted an experiment:

```python
hidden_state = F.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
)

hidden_state_mask1 = F.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False  # Without is_causal
)
difference = torch.abs(hidden_state_mask1 - hidden_state).sum().item()
print("Difference between is_causal=True and bidirectional attention:", difference)
```

The results showed a **significant difference**, indicating that setting `is_causal=True` successfully **converted Bidirectional Attention into Causal Attention**.

------

## 2. LoRA Adaptation Issues and Fixes

During model fine-tuning, we applied **LoRA (Low-Rank Adaptation)** to efficiently adapt CogVideoX. However, we encountered **weight loading failures** when using the fine-tuned model.

### 2.1 **Missing `adapter_config.json`**

After LoRA fine-tuning, the checkpoint **did not automatically generate `adapter_config.json`**, causing `PeftModel.from_pretrained` to fail when loading LoRA weights.

#### **Solution: Using `prepare_adapter.py`**

We manually created a LoRA configuration and generated the `adapter_config.json` file:

```python
transformer_lora_config = LoraConfig(
    r=128,
    lora_alpha=64,
    init_lora_weights=True,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"]
)
```

------

### 2.2 `PeftModel.from_pretrained` Cannot Load Local LoRA Weights

**Issue:**
 By default, `PeftModel.from_pretrained` **attempts to fetch the model from the Hugging Face repository**, rather than loading locally fine-tuned `lora_weights`. This led to **unnecessary network requests** and failed to use the fine-tuned model.

#### **Solution: Manually Loading LoRA Adapters**

By referring to `inference/cli_demo.py`, I manually loaded the local LoRA adapter, bypassing the default online checking logic of `from_pretrained`:

```python
if lora_path:
  pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
  pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)
```

This ensures that `PeftModel` directly loads local LoRA weights, avoiding unnecessary network requests.
