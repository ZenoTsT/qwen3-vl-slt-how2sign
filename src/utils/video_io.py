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
            - "fps2_max32":   sample frames at approximately 2 fps, with a maximum of
                            32 frames per video.

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
        indices = list(range(n_frames)) # return all frame indices [0, 1, 2, ..., n_frames-1]
    else:
        # Here we implement multiple selection strategies.
        if strategy == "uniform":
            indices = np.linspace(0, n_frames - 1, n_frames_to_take, dtype=int).tolist() # creates `n_frames_to_take` integers spaced as evenly as possible
            
        elif strategy == "consecutive":
            indices = list(range(n_frames_to_take)) # Simply take the first `n_frames_to_take` frames: [0, 1, ..., N-1]
        
        elif strategy == "fps2_max32":
            # Sample frames at ~2 fps with an upper bound of 32 frames per video.
            max_frames = 32

            # Try to read FPS from metadata
            try:
                meta = reader.get_meta_data()
                fps = float(meta.get("fps", 0.0) or 0.0)
            except Exception:
                fps = 0.0

            if fps <= 0.0:
                # Fallback: behave roughly like uniform sampling capped at max_frames
                if n_frames <= max_frames:
                    indices = list(range(n_frames))
                else:
                    step = max(n_frames // max_frames, 1)
                    indices = list(range(0, min(n_frames, step * max_frames), step))
            else:
                # Step in frames to get about 2 frames per second
                step = max(fps / 2.0, 1.0)  # frames per sample
                raw_indices = np.arange(0, n_frames, step, dtype=float)
                # Convert to int, unique and sorted
                indices = sorted({int(i) for i in raw_indices if 0 <= int(i) < n_frames})

                # Cap to max_frames
                if len(indices) > max_frames:
                    indices = indices[:max_frames]

                # Safety: at least one frame
                if len(indices) == 0:
                    indices = [0]
            
        else:
            reader.close()
            raise ValueError(f"Unknown strategy: {strategy}")

    # Type annotation: `frames_pil: List[Image.Image]` tells both the type
    # checker and the reader that `frames_pil` is a list of PIL.Image objects.
    frames_pil: List[Image.Image] = []

    # If we executed the fallback above, we pre-loaded all frames into the local variable `frames_raw`. 
    # I check `"frames_raw" in locals()`to know whether that variable exists in the current function scope or not.
    if frames_raw is not None:
        # We just index into that list.
        for idx in indices:
            frame = frames_raw[idx]  # numpy array (H, W, 3) - single frame
            img = Image.fromarray(frame).convert("RGB") # Convert numpy array to PIL image in RGB format
            frames_pil.append(img) # Append the PIL image to the list
    else:
        # Normal path: we still use the `reader` to fetch specific frames one by one with `get_data(idx)`.
        for idx in indices:
            frame = reader.get_data(idx) 
            img = Image.fromarray(frame).convert("RGB")
            frames_pil.append(img)

    reader.close()
    return frames_pil