import pytest
import numpy as np
from fish_detection.feature_extraction import select_best_window, filter_videos, save_features_npz


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


class TestFilterVideos:
    def test_excludes_videos_with_multi_fish(self):
        """Videos with multi-fish frames are excluded."""
        frames1 = [
            {"frame": i, "probability": None, "segment": 1}
            for i in range(20)
        ]
        frames2 = [
            {"frame": i, "probability": None, "segment": 1}
            for i in range(20)
        ]
        detection = {
            "vid1.mp4": {
                "fish_frames": frames1,
                "segments_summary": {"total_segments": 1, "segments": [
                    {"segment_number": 1, "start_frame": 0, "end_frame": 19, "size": 20}
                ]},
            },
            "vid2.mp4": {
                "fish_frames": frames2,
                "segments_summary": {"total_segments": 1, "segments": [
                    {"segment_number": 1, "start_frame": 0, "end_frame": 19, "size": 20}
                ]},
            },
        }
        multi_fish = {
            "vid1.mp4": {"multi_fish_frames": []},
            "vid2.mp4": {"multi_fish_frames": [{"frame": 5}]},
        }
        result = filter_videos(detection, multi_fish)
        assert list(result.keys()) == ["vid1.mp4"]

    def test_excludes_segments_shorter_than_11(self):
        """Segments with fewer than 11 frames are excluded."""
        detection = {
            "vid1.mp4": {
                "fish_frames": [
                    {"frame": i, "probability": 0.99 if i % 3 == 0 else None, "segment": 1}
                    for i in range(5)
                ],
                "segments_summary": {"total_segments": 1, "segments": [
                    {"segment_number": 1, "start_frame": 0, "end_frame": 4, "size": 5}
                ]},
            },
        }
        multi_fish = {"vid1.mp4": {"multi_fish_frames": []}}
        result = filter_videos(detection, multi_fish)
        assert result == {}

    def test_keeps_valid_segments(self):
        """Videos with valid segments (no multi-fish, >= 11 frames) are kept."""
        frames = [
            {"frame": i, "probability": 0.99 if i % 3 == 0 else None, "segment": 1}
            for i in range(20)
        ]
        detection = {
            "vid1.mp4": {
                "fish_frames": frames,
                "segments_summary": {"total_segments": 1, "segments": [
                    {"segment_number": 1, "start_frame": 0, "end_frame": 19, "size": 20}
                ]},
            },
        }
        multi_fish = {"vid1.mp4": {"multi_fish_frames": []}}
        result = filter_videos(detection, multi_fish)
        assert "vid1.mp4" in result
        assert len(result["vid1.mp4"]) == 1


class TestSaveFeatures:
    def test_saves_correct_format(self, tmp_path):
        """NPZ matches training data format."""
        features = {
            10 + i: np.random.randn(1152).astype(np.float32) for i in range(11)
        }
        frame_numbers = list(range(10, 21))
        output_path = tmp_path / "test_features.npz"

        save_features_npz(features, frame_numbers, str(output_path))

        data = np.load(str(output_path), allow_pickle=True)
        assert set(data.keys()) == {
            "features", "frame_numbers", "middle_frame",
            "averaged_features", "fish_species",
        }
        assert data["frame_numbers"].tolist() == frame_numbers
        assert int(data["middle_frame"]) == 15  # frame_numbers[5]
        assert data["averaged_features"].shape == (1152,)
        assert str(data["fish_species"]) == ""

        # Check features dict
        feat_dict = data["features"].item()
        assert len(feat_dict) == 11
        assert all(feat_dict[k].shape == (1152,) for k in feat_dict)
