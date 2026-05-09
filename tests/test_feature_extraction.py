import pytest
import numpy as np
from unittest.mock import MagicMock
from fish_detection.feature_extraction import select_best_window, filter_videos, save_features_npz, classify_date_folder


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
    def test_keeps_segment_with_few_multi_fish(self):
        """A few scattered multi-fish frames (< 11 consecutive) don't exclude the segment."""
        frames = [
            {"frame": i, "probability": None, "segment": 1}
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
        multi_fish = {
            "vid1.mp4": {"multi_fish_frames": [{"frame": 5}, {"frame": 10}]},
        }
        result = filter_videos(detection, multi_fish)
        assert "vid1.mp4" in result

    def test_excludes_segment_overlapping_long_multi_fish_run(self):
        """Segment overlapping with >= 11 consecutive multi-fish frames is excluded."""
        frames = [
            {"frame": i, "probability": None, "segment": 1}
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
        # 11 consecutive multi-fish frames overlapping with segment
        multi_fish = {
            "vid1.mp4": {"multi_fish_frames": [{"frame": i} for i in range(5, 16)]},
        }
        result = filter_videos(detection, multi_fish)
        assert result == {}

    def test_keeps_non_overlapping_segment(self):
        """Segment that doesn't overlap with a multi-fish run is kept."""
        frames_seg1 = [
            {"frame": i, "probability": None, "segment": 1}
            for i in range(20)
        ]
        frames_seg2 = [
            {"frame": i, "probability": None, "segment": 2}
            for i in range(50, 70)
        ]
        detection = {
            "vid1.mp4": {
                "fish_frames": frames_seg1 + frames_seg2,
                "segments_summary": {"total_segments": 2, "segments": [
                    {"segment_number": 1, "start_frame": 0, "end_frame": 19, "size": 20},
                    {"segment_number": 2, "start_frame": 50, "end_frame": 69, "size": 20},
                ]},
            },
        }
        # Multi-fish run overlaps only segment 1
        multi_fish = {
            "vid1.mp4": {"multi_fish_frames": [{"frame": i} for i in range(5, 16)]},
        }
        result = filter_videos(detection, multi_fish)
        assert "vid1.mp4" in result
        assert len(result["vid1.mp4"]) == 1  # only segment 2 kept

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


class TestClassifyDateFolder:
    def test_classifies_and_updates_npz(self, tmp_path):
        """Loads NPZ, predicts species, re-saves with fish_species set."""
        features_dir = tmp_path / "06_08_2025"
        features_dir.mkdir()
        avg = np.random.randn(1152).astype(np.float32)
        npz_path = features_dir / "vid1_seg1_features.npz"
        np.savez(
            str(npz_path),
            features=np.array({}, dtype=object),
            frame_numbers=np.arange(10, 21, dtype=np.int64),
            middle_frame=np.int64(15),
            averaged_features=avg,
            fish_species=np.str_(""),
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(["Lax"])

        classify_date_folder(
            "06_08_2025",
            model=mock_model,
            features_base_dir=str(tmp_path),
        )

        data = np.load(str(npz_path), allow_pickle=True)
        assert str(data["fish_species"]) == "Lax"
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args[0][0]
        assert call_args.shape == (1, 1152)

    def test_skips_already_classified(self, tmp_path):
        """Files with non-empty fish_species are skipped."""
        features_dir = tmp_path / "06_08_2025"
        features_dir.mkdir()
        avg = np.random.randn(1152).astype(np.float32)
        npz_path = features_dir / "vid1_seg1_features.npz"
        np.savez(
            str(npz_path),
            features=np.array({}, dtype=object),
            frame_numbers=np.arange(10, 21, dtype=np.int64),
            middle_frame=np.int64(15),
            averaged_features=avg,
            fish_species=np.str_("Urriði"),
        )

        mock_model = MagicMock()

        classify_date_folder(
            "06_08_2025",
            model=mock_model,
            features_base_dir=str(tmp_path),
        )

        mock_model.predict.assert_not_called()
