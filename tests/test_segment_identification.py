import sys
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import
sys.path.insert(0, str(Path(__file__).parent.parent / "fish_detection"))

from segment_utils import find_segments, fill_intermediate_frames


# Use fps=10 as reference (typical for this project)
# gap_threshold = 2 * fps = 20
# min_segment_length = fps = 10


def test_single_segment():
    """Consecutive detections (every 3 frames) form one segment."""
    fish_frames = [
        {"frame": 99, "probability": 0.99},
        {"frame": 102, "probability": 0.98},
        {"frame": 105, "probability": 0.97},
        {"frame": 108, "probability": 0.99},
        {"frame": 111, "probability": 0.98},
    ]
    # 5 detections, filled = 13 frames (99-111), above min=10
    result = find_segments(fish_frames, min_segment_length=10, gap_threshold=20)
    assert len(result) == 1
    assert result[0]["segment_number"] == 1
    assert result[0]["start_frame"] == 99
    assert result[0]["end_frame"] == 111


def test_two_segments_split_by_gap():
    """Detections separated by >2 seconds (>20 frames) form two segments."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        {"frame": 16, "probability": 0.97},
        {"frame": 19, "probability": 0.99},
        # gap of 31 frames (>20) = new segment
        {"frame": 50, "probability": 0.98},
        {"frame": 53, "probability": 0.97},
        {"frame": 56, "probability": 0.99},
        {"frame": 59, "probability": 0.98},
    ]
    # Each group: 4 detections = 10 filled frames, meets min=10
    result = find_segments(fish_frames, min_segment_length=10, gap_threshold=20)
    assert len(result) == 2
    assert result[0]["segment_number"] == 1
    assert result[0]["start_frame"] == 10
    assert result[1]["segment_number"] == 2
    assert result[1]["start_frame"] == 50


def test_gap_within_threshold_stays_single_segment():
    """Detections with gap <= 20 frames stay in the same segment."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        # gap of 17 frames (<=20), same segment
        {"frame": 30, "probability": 0.97},
        {"frame": 33, "probability": 0.99},
    ]
    result = find_segments(fish_frames, min_segment_length=10, gap_threshold=20)
    assert len(result) == 1
    assert result[0]["start_frame"] == 10
    assert result[0]["end_frame"] == 33


def test_segment_too_short_discarded():
    """Segments shorter than 1 second (< fps frames) are discarded."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        # Only 2 detections = 4 filled frames (10,11,12,13), below min=10
    ]
    result = find_segments(fish_frames, min_segment_length=10, gap_threshold=20)
    assert len(result) == 0


def test_four_detections_valid_segment():
    """4 detections (every 3rd frame) = 10 filled frames = valid at fps=10."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        {"frame": 16, "probability": 0.97},
        {"frame": 19, "probability": 0.99},
    ]
    # 4 detections: frames 10-19 = 10 filled frames, meets min=10
    result = find_segments(fish_frames, min_segment_length=10, gap_threshold=20)
    assert len(result) == 1
    assert result[0]["size"] == 10


def test_empty_input():
    """Empty input returns empty list."""
    result = find_segments([], min_segment_length=10, gap_threshold=20)
    assert result == []


def test_fill_intermediate_frames():
    """Intermediate frames between detections are filled with probability null."""
    fish_frames = [
        {"frame": 99, "probability": 0.99},
        {"frame": 102, "probability": 0.98},
    ]
    filled = fill_intermediate_frames(fish_frames)
    assert len(filled) == 4  # 99, 100, 101, 102
    assert filled[0] == {"frame": 99, "probability": 0.99}
    assert filled[1] == {"frame": 100, "probability": None}
    assert filled[2] == {"frame": 101, "probability": None}
    assert filled[3] == {"frame": 102, "probability": 0.98}


def test_segment_frames_have_segment_number():
    """Each frame in a segment has the segment number assigned."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        {"frame": 16, "probability": 0.97},
        {"frame": 19, "probability": 0.99},
    ]
    result = find_segments(fish_frames, min_segment_length=5, gap_threshold=20)
    assert len(result) == 1
    for frame in result[0]["frames"]:
        assert frame["segment"] == 1
