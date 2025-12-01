# scripts/smoke_test_qwen3vl_all.py

import os
import time
from typing import List, Optional

import torch
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.qwen3vl_lora import load_qwen3vl_lora
from src.utils.video_io import extract_frames_from_video



# ---------------------------------------------------------------------
# 1) SINGLE FRAME SMOKE TEST
# ---------------------------------------------------------------------

def run_single_frame_smoke(model, processor, image_path: str, max_new_tokens: int = 64):
    print("\n================ SINGLE FRAME SMOKE TEST ================")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert("RGB")

    prompt = "You are a vision-language model. Describe what you see in this frame."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=image,               ########## SE NON WORKA RIMETTO LE QUADRE!!!!!!!!!!!!!
        return_tensors="pt",
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    t1 = time.perf_counter()

    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]

    print(f"[TIMER] Single-frame generation time: {t1 - t0:.2f} seconds.")
    print("\n[OUTPUT - SINGLE FRAME]")
    print(generated_text)
    print("=========================================================\n")


# ---------------------------------------------------------------------
# 2) MULTIFRAME (FRAMES DA VIDEO) SMOKE TEST
# ---------------------------------------------------------------------

def run_multiframe_smoke(
    model,
    processor,
    video_path: str,
    num_frames: int = -1,
    strategy: str = "uniform",
    max_new_tokens: int = 64,
):
    print("\n================ MULTIFRAME SMOKE TEST =================")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")

    # Numero di frame: -1 => tutti
    if num_frames == -1:
        n_frames_to_take: Optional[int] = None
    else:
        n_frames_to_take = num_frames

    print(f"[INFO] Extracting frames from video: {video_path}")
    tf0 = time.perf_counter()
    frames: List[Image.Image] = extract_frames_from_video(
        video_path=video_path,
        n_frames_to_take=n_frames_to_take,
        strategy=strategy,
    )
    tf1 = time.perf_counter()
    print(f"[INFO] Extracted {len(frames)} frames.")
    print(f"[TIMER] Frame extraction time: {tf1 - tf0:.2f} seconds.")

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from the video, cannot run multiframe smoke test.")

    prompt = (
        "You are a vision-language model. Given this short video, "
        "describe what is happening in the scene."
    )

    content_blocks = [{"type": "image"} for _ in range(len(frames))]
    content_blocks.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content_blocks,
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prep_t0 = time.perf_counter()
    inputs = processor(
        text=[chat_text],
        images=frames,
        return_tensors="pt",
    ).to(model.device)
    prep_t1 = time.perf_counter()
    print(f"[TIMER] Input preparation (multiframe) time: {prep_t1 - prep_t0:.2f} seconds.")

    print("[INFO] Running multiframe generation...")
    gen_t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_t1 = time.perf_counter()
    print(f"[TIMER] Multiframe generation time: {gen_t1 - gen_t0:.2f} seconds.")

    decoded_t0 = time.perf_counter()
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]
    decoded_t1 = time.perf_counter()
    print(f"[TIMER] Multiframe decoding time: {decoded_t1 - decoded_t0:.2f} seconds.")

    print("\n[OUTPUT - MULTIFRAME]")
    print(generated_text)
    print("=========================================================\n")


# ---------------------------------------------------------------------
# 3) VIDEO SMOKE TEST (API VIDEO DIRETTA)
# ---------------------------------------------------------------------

def run_video_smoke(
    model,
    processor,
    video_path: str,
    max_new_tokens: int = 64,
):
    print("\n=================== VIDEO SMOKE TEST ====================")

    prompt = (
        "You are a vision-language model. Given this short video, "
        "describe what is happening in the scene."
    )

    # Messages: usa direttamente il video_path
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,              
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    chat_t1 = time.perf_counter()
    print(f"[TIMER] apply_chat_template(): {chat_t1 - chat_t0:.2f} seconds.")

    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    print("[INFO] Running generation on VIDEO...")
    gen_t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_t1 = time.perf_counter()
    print(f"[TIMER] Video generation time: {gen_t1 - gen_t0:.2f} seconds.")

    generated_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("\n[OUTPUT - VIDEO]")
    print(generated_text)
    print("=========================================================\n")

# ---------------------------------------------------------------------
# 4) VIDEO SMOKE TEST CUSTOM (MANDO MANUALMENTE IO I FRAME)
# ---------------------------------------------------------------------

def run_video_smoke_custom(
    model,
    processor,
    video_path: str,
    max_new_tokens: int = 64,
    num_frames: int = -1,
    strategy: str = "uniform",
):
    """
    Smoke test per l'inferenza video con Qwen3-VL.

    Differenza fondamentale rispetto a prima:
    - estraiamo i frame
    - li passiamo come `videos=[frames]` al processor
    - in `messages` usiamo solo {"type": "video"} come placeholder
    """

    print("\n=================== VIDEO SMOKE TEST (CUSTOM) ====================")

    # 1) Estrazione frame dal video
    print(f"[INFO] Extracting frames for video mode from: {video_path}")
    tf0 = time.perf_counter()
    frames = extract_frames_from_video(
        video_path=video_path,
        n_frames_to_take=num_frames if num_frames > 0 else None,
        strategy=strategy,
    )
    tf1 = time.perf_counter()
    print(f"[INFO] Extracted {len(frames)} frames for VIDEO.")
    print(f"[TIMER] Frame extraction (video) time: {tf1 - tf0:.2f} seconds.")

    if len(frames) == 0:
        raise RuntimeError("No frames extracted for video inference.")

    # 2) Prompt
    prompt = (
        "You are a vision-language model. Given this short video, "
        "describe what is happening in the scene."
    )

    # 3) Messages: usiamo un singolo blocco {"type": "video"}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},  # placeholder; i pixel veri arrivano via videos=[frames]
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 4) apply_chat_template genera una stringa di prompt dall oggetto messages (applicando token speciali)
    chat_t0 = time.perf_counter()
    chat_text = processor.apply_chat_template( 
        messages,
        tokenize=False,                     # Non rende il testo subito tokenizzato (in quanto poi dobbiamo ancora passare pure il video)
        add_generation_prompt=True,
    )
    chat_t1 = time.perf_counter()
    print(f"[TIMER] apply_chat_template() (video): {chat_t1 - chat_t0:.2f} seconds.")

    # 5) Utilizziamo il processor di qwen per passare sia il messaggio di prima, che il video
    prep_t0 = time.perf_counter()
    inputs = processor(                     
        text=[chat_text],
        videos=[frames],      
        return_tensors="pt",
    ).to(model.device)
    prep_t1 = time.perf_counter()
    print(f"[TIMER] Input preparation (video) time: {prep_t1 - prep_t0:.2f} seconds.")

    # 6) Generazione
    print("[INFO] Running generation on VIDEO...")
    gen_t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # deterministico
        )
    gen_t1 = time.perf_counter()
    print(f"[TIMER] Video generation time: {gen_t1 - gen_t0:.2f} seconds.")

    # 7) Decodifica
    dec_t0 = time.perf_counter()
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]
    dec_t1 = time.perf_counter()
    print(f"[TIMER] Video decoding time: {dec_t1 - dec_t0:.2f} seconds.\n")

    print("[OUTPUT - VIDEO]")
    print(generated_text)
    print("=========================================================\n")


# ---------------------------------------------------------------------
# MAIN: carica una sola volta, poi chiama i 4 smoke test
# ---------------------------------------------------------------------

def main():
    script_t0 = time.perf_counter()

    # 1) Load model + processor una sola volta
    print("[INFO] Loading model + processor with LoRA...")
    load_t0 = time.perf_counter()
    model, processor = load_qwen3vl_lora(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        r=4,         # small LoRA rank for smoke tests
        alpha=8,
        dropout=0.1,
        device_map="auto",
    )
    model.eval()
    load_t1 = time.perf_counter()
    print(f"[TIMER] load_qwen3vl_lora(): {load_t1 - load_t0:.2f} seconds.")
    print(f"[INFO] Model device: {model.device}")

    # 2) Path di test (puoi cambiarli se vuoi)
    image_path = "tests/assets/example.jpg"
    video_path = "tests/assets/_fZbAxSSbX4_0-5-rgb_front.mp4"

    # 3) Lancia i 4 smoke test in sequenza
    run_single_frame_smoke(
        model=model,
        processor=processor,
        image_path=image_path,
        max_new_tokens=64,
    )

    # run_multiframe_smoke(
    #     model=model,
    #     processor=processor,
    #     video_path=video_path,
    #     num_frames=-1,      # -1 => tutti i frame
    #     strategy="uniform",
    #     max_new_tokens=64,
    # )

    run_video_smoke(
        model=model,
        processor=processor,
        video_path=video_path,
        max_new_tokens=64,
    )
    
    run_video_smoke_custom(
        model=model,
        processor=processor,
        video_path=video_path,
        max_new_tokens=64,
    )

    script_t1 = time.perf_counter()
    print(f"[TIMER] Total script time: {script_t1 - script_t0:.2f} seconds.")


if __name__ == "__main__":
    main()