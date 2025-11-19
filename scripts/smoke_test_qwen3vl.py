import torch
from PIL import Image

from src.models.qwen3vl_lora import load_qwen3vl_lora


def main():
    # 1) Load model + processor with small LoRA config for testing
    model, processor = load_qwen3vl_lora(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        r=4,         # very small LoRA rank just for smoke test
        alpha=8,
        dropout=0.1,
        device_map="auto",
    )

    model.eval()  # disable dropout, put model in eval mode

    # 2) Load a test image (put any jpg/png here)
    image_path = "tests/assets/example.jpg"  # change this path if needed
    image = Image.open(image_path).convert("RGB")

    # 3) Define a simple prompt
    prompt = "You are a vision-language model. Describe what you see in this frame."

    # 4) Use the processor to prepare multimodal inputs (image + text)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Applica il chat template -> inserisce automaticamente i token immagine
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=[chat_text],    # lista di stringhe
        images=[image],      # lista di immagini
        return_tensors="pt",
    ).to(model.device)

    # 5) Generate a response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    # 6) Decode the generated text
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]

    print("Generated:", generated_text)


if __name__ == "__main__":
    main()