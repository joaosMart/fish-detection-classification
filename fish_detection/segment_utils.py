"""Segment identification utilities for fish detection pipeline."""

import numpy as np
from typing import List, Dict


def fill_intermediate_frames(fish_frames: List[Dict]) -> List[Dict]:
    """Fill in frames between detections with probability: None.

    Given detected frames (e.g., 99, 102, 105), inserts intermediate
    frames (100, 101, 103, 104) with probability set to None.
    """
    if not fish_frames:
        return []

    fish_frames_sorted = sorted(fish_frames, key=lambda x: x['frame'])
    filled = []

    for i, frame_data in enumerate(fish_frames_sorted):
        filled.append(frame_data.copy())

        if i < len(fish_frames_sorted) - 1:
            current_frame = frame_data['frame']
            next_frame = fish_frames_sorted[i + 1]['frame']
            # Fill intermediate frames
            for intermediate in range(current_frame + 1, next_frame):
                filled.append({"frame": intermediate, "probability": None})

    return filled


def find_segments(fish_frames: List[Dict],
                  min_segment_length: int = 11,
                  gap_threshold: int = 11) -> List[Dict]:
    """Find all valid segments in a list of fish frame detections.

    A segment is a group of consecutive detections where no gap between
    adjacent detected frames exceeds gap_threshold frame numbers.

    Intermediate frames are filled in and segments shorter than
    min_segment_length (after filling) are discarded.

    Args:
        fish_frames: List of dicts with 'frame' and 'probability' keys
        min_segment_length: Minimum frames (with intermediates) for a valid segment
        gap_threshold: Maximum gap in frame numbers before splitting into new segment

    Returns:
        List of segment dicts with segment_number, start_frame, end_frame, size, frames
    """
    if not fish_frames:
        return []

    # Sort by frame number
    sorted_frames = sorted(fish_frames, key=lambda x: x['frame'])
    frame_numbers = [f['frame'] for f in sorted_frames]

    # Find gaps exceeding threshold
    gaps = np.diff(frame_numbers)
    split_indices = np.where(gaps > gap_threshold)[0]

    # Build segment boundaries
    boundaries = []
    start_idx = 0
    for idx in split_indices:
        boundaries.append((start_idx, idx + 1))
        start_idx = idx + 1
    boundaries.append((start_idx, len(frame_numbers)))

    # Process each candidate segment
    segments = []
    segment_number = 1

    for start_idx, end_idx in boundaries:
        segment_detections = sorted_frames[start_idx:end_idx]

        # Fill intermediate frames
        filled_frames = fill_intermediate_frames(segment_detections)

        # Check minimum length
        if len(filled_frames) >= min_segment_length:
            # Assign segment number to each frame
            for frame in filled_frames:
                frame['segment'] = segment_number

            segments.append({
                'segment_number': segment_number,
                'start_frame': filled_frames[0]['frame'],
                'end_frame': filled_frames[-1]['frame'],
                'size': len(filled_frames),
                'frames': filled_frames,
            })
            segment_number += 1

    return segments
