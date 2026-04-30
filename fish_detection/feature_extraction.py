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


def filter_videos(
    detection_results: Dict[str, Dict],
    multi_fish_results: Dict[str, Dict],
    min_segment_size: int = 11,
) -> Dict[str, List[List[Dict]]]:
    """Filter videos to those with valid single-fish segments.

    Excludes videos with any multi-fish frames. For remaining videos,
    returns only segments with at least min_segment_size frames.

    Args:
        detection_results: Single-fish detection results keyed by video path.
        multi_fish_results: Multi-fish detection results keyed by video path.
        min_segment_size: Minimum segment size to keep.

    Returns:
        Dict mapping video_path -> list of segment frame lists.
        Each segment frame list is sorted by frame number and contains
        dicts with 'frame' and 'probability' keys.
    """
    valid = {}

    for video_path, det in detection_results.items():
        # Skip if video has multi-fish frames
        multi = multi_fish_results.get(video_path, {})
        if multi.get("multi_fish_frames"):
            continue

        # Group fish_frames by segment number
        segments_by_num: Dict[int, List[Dict]] = {}
        for frame in det["fish_frames"]:
            seg_num = frame.get("segment")
            if seg_num is None:
                continue
            segments_by_num.setdefault(seg_num, []).append(frame)

        # Filter segments by minimum size
        valid_segments = []
        for seg_num in sorted(segments_by_num.keys()):
            seg_frames = sorted(segments_by_num[seg_num], key=lambda f: f["frame"])
            if len(seg_frames) >= min_segment_size:
                valid_segments.append(seg_frames)

        if valid_segments:
            valid[video_path] = valid_segments

    return valid
