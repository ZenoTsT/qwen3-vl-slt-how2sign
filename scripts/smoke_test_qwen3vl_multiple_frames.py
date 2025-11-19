# scripts/smoke_test_qwen3vl_multiframe.py

import argparse
import time
from typing import List, Optional

import torch
from PIL import Image

from src.models.qwen3vl_lora import load_qwen3vl_lora
from src.utils.video_io import extract_frames_from_video


def build_multiframe_messages(num_images: int, prompt: str):
    """
    Build a Qwen3-VL chat-style message with multiple image slots.

    Qwen uses a list of content blocks. Each {"type": "image"} corresponds
    to one image in the `images` list passed to the processor.
    """
    content_blocks = [{"type": "image"} for _ in range(num_images)]
    content_blocks.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": content_blocks,
        }
    ]
    return messages


def parse_args():
    """
    Parse command-line arguments using argparse.

    Example:
        python smoke_test_qwen3vl_multiframe.py \
            --video path/to/video.mp4 \
            --num_frames 8
    """
    parser = argparse.ArgumentParser(description="Qwen3-VL multiframe smoke test")

    parser.add_argument(
        "--video",
        type=str,
        default="tests/assets/_fZbAxSSbX4_0-5-rgb_front.mp4",
        help="Path to the input video file.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help=(
            "Number of frames to sample from the video. "
            "If -1 (default), all frames are used."
        ),
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="uniform",
        choices=["uniform", "consecutive", "center", "random"],
        help="Frame sampling strategy used inside extract_frames_from_video.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate.",
    )

    return parser.parse_args()


def main():
    script_t0 = time.perf_counter()

    args = parse_args()

    # ---------------------------------------------------
    # 1) Load model + processor with small LoRA for smoke test
    # ---------------------------------------------------
    load_t0 = time.perf_counter()
    model, processor = load_qwen3vl_lora(
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        r=4,          # very small LoRA rank just for smoke test
        alpha=8,
        dropout=0.1,
        device_map="auto",
    )
    model.eval()  # disable dropout, evaluation mode
    load_t1 = time.perf_counter()
    print(f"[TIMER] load_qwen3vl_lora() inside script: {load_t1 - load_t0:.2f} seconds.")

    # ---------------------------------------------------
    # 2) Extract frames from video
    # ---------------------------------------------------
    if args.num_frames == -1:
        n_frames_to_take: Optional[int] = None  # None -> all frames
    else:
        n_frames_to_take = args.num_frames

    print(f"[INFO] Extracting frames from video: {args.video}")
    frames_t0 = time.perf_counter()
    frames: List[Image.Image] = extract_frames_from_video(
        video_path=args.video,
        n_frames_to_take=n_frames_to_take,
        strategy=args.strategy,
    )
    frames_t1 = time.perf_counter()
    print(f"[INFO] Extracted {len(frames)} frames.")
    print(f"[TIMER] Frame extraction time: {frames_t1 - frames_t0:.2f} seconds.")

    if len(frames) == 0:
        raise RuntimeError("No frames extracted from the video, cannot run smoke test.")

    # ---------------------------------------------------
    # 3) Build prompt and chat template with multiple images
    # ---------------------------------------------------
    prompt = (
        "You are a vision-language model. Given these video frames, "
        "describe what is happening in the scene."
    )

    messages = build_multiframe_messages(num_images=len(frames), prompt=prompt)

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ---------------------------------------------------
    # 4) Prepare multimodal inputs for the model
    # ---------------------------------------------------
    prep_t0 = time.perf_counter()
    inputs = processor(
        text=[chat_text],
        images=frames,
        return_tensors="pt",
    ).to(model.device)
    prep_t1 = time.perf_counter()
    print(f"[TIMER] Input preparation (processor) time: {prep_t1 - prep_t0:.2f} seconds.")

    # ---------------------------------------------------
    # 5) Generation
    # ---------------------------------------------------
    print("[INFO] Running generation...")
    gen_t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # deterministic for smoke test
        )
    gen_t1 = time.perf_counter()
    print(f"[TIMER] Generation time: {gen_t1 - gen_t0:.2f} seconds.")

    # ---------------------------------------------------
    # 6) Decode and print
    # ---------------------------------------------------
    decoded_t0 = time.perf_counter()
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]
    decoded_t1 = time.perf_counter()
    print(f"[TIMER] Decoding time: {decoded_t1 - decoded_t0:.2f} seconds.")

    print("\n===== MULTIFRAME SMOKE TEST OUTPUT =====")
    print(generated_text)
    print("========================================\n")

    script_t1 = time.perf_counter()
    print(f"[TIMER] Total script time: {script_t1 - script_t0:.2f} seconds.")


if __name__ == "__main__":
    main()