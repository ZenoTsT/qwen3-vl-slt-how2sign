import torch
from transformers import Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

def is_linear(m):
    return isinstance(m, torch.nn.Linear)

groups = {
    "vision": [],
    "vision_merger": [],
    "language": [],
    "other": [],
}

for name, module in model.named_modules():
    if not is_linear(module):
        continue

    if name.startswith("model.visual.blocks"):
        groups["vision"].append(name)
    elif "visual.merger" in name or "deepstack_merger" in name:
        groups["vision_merger"].append(name)
    elif name.startswith("model.language_model"):
        groups["language"].append(name)
    else:
        groups["other"].append(name)

for k, v in groups.items():
    print(f"\n==== {k.upper()} ({len(v)}) ====")
    for n in v:
        print(n)