# src/utils/video_io.py

import os
from typing import List, Optional

import numpy as np
from PIL import Image
import imageio.v2 as imageio


def extract_frames_from_video(
    video_path: str,
    n_frames_to_take: Optional[int] = None,
    strategy: str = "uniform",
    seed: Optional[int] = None,
) -> List[Image.Image]:
    """
    Extract frames from a video and return them as a list of RGB PIL.Image objects.

    Args:
        video_path:
            Path to the video file.
        n_frames_to_take:
            - If None -> take *all* frames from the video.
            - If an integer -> select exactly `n_frames_to_take` frames using the
              specified `strategy`.
        strategy:
            Strategy used to select which frames to keep (when n_frames_to_take
            is not None and smaller than the total number of frames):
            - "uniform":    frames are sampled uniformly across the whole video
                           (e.g., for 100 frames and n_frames_to_take=10, we
                           roughly take one every 10 frames).
            - "consecutive": take the first `n_frames_to_take` consecutive frames
                            from the beginning of the video.
            - "center":     take `n_frames_to_take` consecutive frames centered
                            around the middle of the video.
            - "random":     pick `n_frames_to_take` random frames (without
                            replacement). If `seed` is provided, the sampling
                            is reproducible.
        seed:
            Optional random seed used only when strategy="random".

    Returns:
        List[Image.Image]: list of frames as RGB images in temporal order.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # `reader` is a video reader object provided by imageio. It uses the "ffmpeg" backend to decode the video file
    reader = imageio.get_reader(video_path, "ffmpeg")

    # Count the total number of frames in the video.
    try:
        n_frames = reader.count_frames()
        frames_raw = None  # we did not pre-load frames here
    except Exception:
        # Fallback: if `count_frames()` is not implemented or fails, we read ALL frames into memory as a Python list.
        frames_raw = list(reader)
        n_frames = len(frames_raw)

    if n_frames == 0:
        reader.close()
        raise ValueError(f"No frames found in video: {video_path}")

    # Normalize n_frames_to_take: if None or too large, just take everything.
    if n_frames_to_take is None or n_frames_to_take >= n_frames:
        indices = list(range(n_frames)) # list of indices
    else:
        # Here we implement multiple selection strategies.
        if strategy == "uniform":
            indices = np.linspace(0, n_frames - 1, n_frames_to_take, dtype=int).tolist() # creates `n_frames_to_take` integers spaced as evenly as possible
            
        elif strategy == "consecutive":
            indices = list(range(n_frames_to_take)) # Simply take the first `n_frames_to_take` frames: [0, 1, ..., N-1]
            
        elif strategy == "center":
            # Choose a consecutive window of size `n_frames_to_take` centered as much as possible around the middle of the video.
            center = (n_frames - 1) / 2.0
            half = (n_frames_to_take - 1) / 2.0
            start = int(round(center - half))
            if start < 0:
                start = 0
            end = start + n_frames_to_take
            if end > n_frames:
                end = n_frames
                start = end - n_frames_to_take
                if start < 0:
                    start = 0
            indices = list(range(start, end))
            
        elif strategy == "random":
            # Randomly sample `n_frames_to_take` distinct frame indices, using a Generator lets us optionally control reproducibility via `seed`.
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(n_frames, size=n_frames_to_take, replace=False).tolist())
            
        else:
            reader.close()
            raise ValueError(f"Unknown strategy: {strategy}")

    # Type annotation: `frames_pil: List[Image.Image]` tells both the type
    # checker and the reader that `frames_pil` is a list of PIL.Image objects.
    frames_pil: List[Image.Image] = []

    # If we executed the fallback above, we pre-loaded all frames into the local variable `frames_raw`. 
    # I check `"frames_raw" in locals()`to know whether that variable exists in the current function scope or not.
    if "frames_raw" in locals() and frames_raw is not None:
        # We just index into that list.
        for idx in indices:
            frame = frames_raw[idx]  # numpy array (H, W, 3) - single frame
            img = Image.fromarray(frame).convert("RGB")
            frames_pil.append(img)
    else:
        # Normal path: we still use the `reader` to fetch specific frames one by one with `get_data(idx)`.
        for idx in indices:
            frame = reader.get_data(idx)  # numpy array (H, W, 3) - single frame
            img = Image.fromarray(frame).convert("RGB")
            frames_pil.append(img)

    reader.close()
    return frames_pil