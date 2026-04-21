# Segment Identification Design

## Purpose

Add segment identification to the fish detection pipeline. A "segment" represents a continuous period where a fish is present in the video. Multiple segments in one video indicate multiple fish passing events separated by gaps.

## Integration

- New `identify_segments` method on `FishDetector`
- Called in `run()` after `filter_results()`, only for single-fish detection mode
- Enriches `results.json` in-place (no separate output file)

## Algorithm

1. For each video's `fish_frames` list (already filtered by detection threshold):
   - Sort by frame number
   - Compute gaps between consecutive detected frames using `np.diff`
   - Split into segments where gap > `gap_threshold` (default: 11 frame numbers)
   - For each candidate segment:
     - Fill intermediate frames between detections (e.g., frames 100, 101 between detections at 99 and 102) with `probability: null`
     - If total segment length (with intermediates) >= `min_segment_length` (default: 11), keep it
     - Assign incrementing `segment` number to each frame in the segment

2. Parameters:
   - `min_segment_length: int = 11` — minimum frames (including intermediates) for a valid segment
   - `gap_threshold: int = 11` — gap in frame numbers that triggers a segment split

## Output Format

The existing `results.json` is enriched. Per video:

```json
{
  "total_frames": 237,
  "frames_processed": 79,
  "fish_frames": [
    {"frame": 99, "probability": 0.985, "segment": 1},
    {"frame": 100, "probability": null, "segment": 1},
    {"frame": 101, "probability": null, "segment": 1},
    {"frame": 102, "probability": 0.999, "segment": 1},
    ...
    {"frame": 200, "probability": 0.991, "segment": 2}
  ],
  "segments_summary": {
    "total_segments": 2,
    "segments": [
      {"segment_number": 1, "start_frame": 99, "end_frame": 150, "size": 52},
      {"segment_number": 2, "start_frame": 200, "end_frame": 230, "size": 31}
    ]
  }
}
```

Videos with no fish frames get `segments_summary: {"total_segments": 0, "segments": []}`.

## Filling Intermediates

Since detection runs on every 3rd frame (0, 3, 6, 9...), consecutive detections are typically 3 frame numbers apart. Between two detected frames N and M (where M - N <= gap_threshold), insert all integer frames N+1, N+2, ..., M-1 with `probability: null`. This provides continuous frame coverage within a segment.

## Scope

- Only applies to single-fish detection mode (`detection_type == "single"`)
- Multi-fish detection does not use segmentation (it operates on pre-filtered fish frames)
- No new CLI mode needed — runs automatically as part of `run()`

## Files Modified

- `fish_detection/fish-detection.py`: Add `identify_segments` method, call it in `run()` after `filter_results()`
