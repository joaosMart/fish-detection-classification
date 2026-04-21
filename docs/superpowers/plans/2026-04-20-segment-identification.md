# Segment Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add segment identification to the single-fish detection pipeline, grouping consecutive fish detections into distinct segments representing individual fish passing events.

**Architecture:** New `identify_segments` method on `FishDetector` called after `filter_results()` in the `run()` flow. It groups fish frames by gap analysis, fills intermediate frames, and enriches `results.json` with segment assignments and summary.

**Tech Stack:** Python, NumPy (already imported)

---

## File Structure

- **Modify:** `fish_detection/fish-detection.py` — add `identify_segments` method, update `run()` to call it
- **Create:** `tests/test_segment_identification.py` — unit tests for segment logic

---

### Task 1: Write Tests for Segment Identification

**Files:**
- Create: `tests/test_segment_identification.py`

- [ ] **Step 1: Create test file with test cases**

```python
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import
sys.path.insert(0, str(Path(__file__).parent.parent / "fish_detection"))

# We'll test the standalone function before integrating into the class
from segment_utils import find_segments, fill_intermediate_frames


def test_single_segment():
    """Consecutive detections (every 3 frames) form one segment."""
    fish_frames = [
        {"frame": 99, "probability": 0.99},
        {"frame": 102, "probability": 0.98},
        {"frame": 105, "probability": 0.97},
        {"frame": 108, "probability": 0.99},
        {"frame": 111, "probability": 0.98},
    ]
    result = find_segments(fish_frames, min_segment_length=11, gap_threshold=11)
    assert len(result) == 1
    assert result[0]["segment_number"] == 1
    # 99 to 111 inclusive = 13 frames (with intermediates filled)
    assert result[0]["start_frame"] == 99
    assert result[0]["end_frame"] == 111


def test_two_segments_split_by_gap():
    """Detections separated by >11 frame numbers form two segments."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        {"frame": 16, "probability": 0.97},
        {"frame": 19, "probability": 0.99},
        # gap of 30 frames here
        {"frame": 50, "probability": 0.98},
        {"frame": 53, "probability": 0.97},
        {"frame": 56, "probability": 0.99},
        {"frame": 59, "probability": 0.98},
    ]
    result = find_segments(fish_frames, min_segment_length=11, gap_threshold=11)
    assert len(result) == 2
    assert result[0]["segment_number"] == 1
    assert result[0]["start_frame"] == 10
    assert result[1]["segment_number"] == 2
    assert result[1]["start_frame"] == 50


def test_segment_too_short_discarded():
    """Segments shorter than min_segment_length are discarded."""
    fish_frames = [
        {"frame": 10, "probability": 0.99},
        {"frame": 13, "probability": 0.98},
        # Only 4 frames (10,11,12,13) — below 11
    ]
    result = find_segments(fish_frames, min_segment_length=11, gap_threshold=11)
    assert len(result) == 0


def test_empty_input():
    """Empty input returns empty list."""
    result = find_segments([], min_segment_length=11, gap_threshold=11)
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
    result = find_segments(fish_frames, min_segment_length=5, gap_threshold=11)
    assert len(result) == 1
    for frame in result[0]["frames"]:
        assert frame["segment"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/joaodsm/Desktop/Joao Workspace/Fish Detection and Classification/fish-detection/fish-detection" && python -m pytest tests/test_segment_identification.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'segment_utils'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_segment_identification.py
git commit -m "test: add segment identification tests"
```

---

### Task 2: Implement Segment Utility Functions

**Files:**
- Create: `fish_detection/segment_utils.py`

- [ ] **Step 1: Create segment_utils.py with core logic**

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd "/Users/joaodsm/Desktop/Joao Workspace/Fish Detection and Classification/fish-detection/fish-detection" && python -m pytest tests/test_segment_identification.py -v`
Expected: All 6 tests PASS

- [ ] **Step 3: Commit**

```bash
git add fish_detection/segment_utils.py
git commit -m "feat: add segment identification utility functions"
```

---

### Task 3: Integrate into FishDetector Pipeline

**Files:**
- Modify: `fish_detection/fish-detection.py:510-527` (the `run` method)

- [ ] **Step 1: Add identify_segments method to FishDetector class**

Add this method after `print_summary` (around line 508), before the `run` method:

```python
def identify_segments(self, filtered_results: Dict[str, Dict],
                      min_segment_length: int = 11,
                      gap_threshold: int = 11) -> Dict[str, Dict]:
    """Identify fish segments in filtered results.

    Groups consecutive fish detections into segments, fills intermediate
    frames, and enriches results with segment assignments.
    """
    from fish_detection.segment_utils import find_segments

    for video_path, data in filtered_results.items():
        fish_frames = data.get('fish_frames', [])

        if not fish_frames:
            data['segments_summary'] = {'total_segments': 0, 'segments': []}
            continue

        segments = find_segments(fish_frames, min_segment_length, gap_threshold)

        # Replace fish_frames with enriched frames from all valid segments
        enriched_frames = []
        segments_summary = []
        for seg in segments:
            enriched_frames.extend(seg['frames'])
            segments_summary.append({
                'segment_number': seg['segment_number'],
                'start_frame': seg['start_frame'],
                'end_frame': seg['end_frame'],
                'size': seg['size'],
            })

        data['fish_frames'] = enriched_frames
        data['segments_summary'] = {
            'total_segments': len(segments),
            'segments': segments_summary,
        }

    # Save enriched results
    with open(self.results_file, 'w') as f:
        json.dump(filtered_results, f, indent=2)

    print(f"Segment identification complete. Results updated in: {self.results_file}")
    return filtered_results
```

- [ ] **Step 2: Update run() to call identify_segments for single detection**

In the `run` method (line ~525), after `results = self.filter_results(scores)`, add:

```python
# Identify segments (single fish detection only)
if self.detection_type == "single":
    results = self.identify_segments(results)
```

- [ ] **Step 3: Run tests to verify nothing is broken**

Run: `cd "/Users/joaodsm/Desktop/Joao Workspace/Fish Detection and Classification/fish-detection/fish-detection" && python -m pytest tests/test_segment_identification.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add fish_detection/fish-detection.py
git commit -m "feat: integrate segment identification into pipeline"
```

---

### Task 4: Manual Verification with Existing Data

**Files:**
- None (verification only)

- [ ] **Step 1: Run segment identification on existing results**

Create a quick verification script to test against the existing `results.json`:

```bash
cd "/Users/joaodsm/Desktop/Joao Workspace/Fish Detection and Classification/fish-detection/fish-detection"
python -c "
import json
import sys
sys.path.insert(0, 'fish_detection')
from segment_utils import find_segments

with open('output/detection_output/14_08_2025/fish_detection/results.json') as f:
    results = json.load(f)

for video, data in results.items():
    fish_frames = data.get('fish_frames', [])
    if fish_frames:
        segments = find_segments(fish_frames)
        print(f'{video}: {len(fish_frames)} detections -> {len(segments)} segments')
        for seg in segments:
            print(f'  Segment {seg[\"segment_number\"]}: frames {seg[\"start_frame\"]}-{seg[\"end_frame\"]} ({seg[\"size\"]} frames)')
"
```

- [ ] **Step 2: Verify output makes sense**

Check that segments correspond to visually distinct fish events (frames with large gaps between them get split).

- [ ] **Step 3: Commit verification notes (optional)**

No code changes needed if verification passes.
