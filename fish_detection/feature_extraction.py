"""Feature extraction for fish species classification pipeline."""

import numpy as np
from typing import List, Dict, Optional


def select_best_window(
    segment_frames: List[Dict],
    window_size: int = 11,
) -> Optional[List[int]]:
    """Select the best window of consecutive frames from a segment.

    Slides a window of `window_size` frames and picks the one where the
    scored frames (non-None probability) have the highest mean probability.

    Args:
        segment_frames: List of dicts with 'frame' and 'probability' keys,
                        sorted by frame number.
        window_size: Number of consecutive frames in the window.

    Returns:
        List of frame numbers for the best window, or None if segment
        is shorter than window_size.
    """
    if len(segment_frames) < window_size:
        return None

    best_start = 0
    best_mean = -1.0

    for start in range(len(segment_frames) - window_size + 1):
        window = segment_frames[start : start + window_size]
        scored = [f["probability"] for f in window if f["probability"] is not None]
        if not scored:
            continue
        mean_prob = sum(scored) / len(scored)
        if mean_prob > best_mean:
            best_mean = mean_prob
            best_start = start

    return [segment_frames[best_start + i]["frame"] for i in range(window_size)]
