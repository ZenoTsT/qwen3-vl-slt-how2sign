import torch
from transformers import AutoModelForCausalLM, AutoProcessor #, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model


def load_qwen3vl_lora(                                      # ΔW ≈ A · B, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d)
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct", 
    r: int = 16,                                            # LoRA rank - dimension of the LoRA matrices
    alpha: int = 32,                                        # scale factor (how much weight to give to LoRA)
    dropout: float = 0.05,                                  # dropout used for LoRA
    device_map: str = "auto",                               # how to map the model to devices/GPUs
):
    """
    Load Qwen3-VL vision-language model with LoRA applied.
    """

    torch_dtype = torch.float16                             # data type for model weights (fp16 to save memory)

    processor = AutoProcessor.from_pretrained(              # load the multimodal processor (tokenizer + image processor, the "vocabulary" and preprocessing of the MLLM)
        model_name, 
        trust_remote_code=True,
    )

    # model = Qwen3VLForConditionalGeneration.from_pretrained(           # load the pretrained multimodal causal LM (text + vision)
    #     model_name,
    #     dtype=torch_dtype,
    #     device_map=device_map,
    #     attn_implementation="sdpa",
    #     #attn_implementation="eager",
    #     trust_remote_code=True,
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # Config LoRA (for now only on attention projections, we will refine this later)
    # We will focus LoRA only on the layers that need it, not on all layers as now
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],    # we apply LoRA to queries, keys, values and outputs of attention
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",                              
    )

    model = get_peft_model(model, lora_config)              # wrap the base model with LoRA adapters

    # Print how many parameters are trainable
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.4f}%)")

    return model, processor