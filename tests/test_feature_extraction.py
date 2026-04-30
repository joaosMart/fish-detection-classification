import pytest
from fish_detection.feature_extraction import select_best_window


class TestSelectBestWindow:
    def test_exact_11_frames_returns_all(self):
        """Segment of exactly 11 frames: only one window position."""
        frames = [
            {"frame": i, "probability": 0.99 if i % 3 == 0 else None}
            for i in range(10, 21)
        ]
        result = select_best_window(frames, window_size=11)
        assert result == list(range(10, 21))

    def test_picks_highest_mean_prob_window(self):
        """Window with highest mean of scored frames wins."""
        frames = []
        for i in range(15):
            if i % 3 == 0:
                prob = 0.99 if i == 12 else 0.5
            else:
                prob = None
            frames.append({"frame": i, "probability": prob})
        result = select_best_window(frames, window_size=11)
        assert result == list(range(4, 15))

    def test_segment_shorter_than_window_returns_none(self):
        """Segment with fewer than 11 frames returns None."""
        frames = [{"frame": i, "probability": 0.9} for i in range(5)]
        result = select_best_window(frames, window_size=11)
        assert result is None

    def test_non_contiguous_frame_numbers(self):
        """Works with non-zero-based frame numbers."""
        frames = [
            {"frame": 42 + i, "probability": 0.95 if (42 + i) % 3 == 0 else None}
            for i in range(11)
        ]
        result = select_best_window(frames, window_size=11)
        assert result == list(range(42, 53))
        assert len(result) == 11

    def test_ties_pick_first_window(self):
        """When multiple windows tie on mean prob, the first one wins."""
        frames = [
            {"frame": i, "probability": 1.0 if i % 3 == 0 else None}
            for i in range(15)
        ]
        result = select_best_window(frames, window_size=11)
        assert result == list(range(0, 11))
