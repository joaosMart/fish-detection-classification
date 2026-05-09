"""Feature extraction for fish species classification pipeline."""

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import cv2
import joblib
from PIL import Image
from pathlib import Path
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


def load_siglip_model(device: torch.device = None):
    """Load the SigLIP model and preprocessing transform.

    Returns:
        Tuple of (model, preprocess, device).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP", pretrained="webli"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess, device


def extract_siglip_features(
    video_path: str,
    frame_numbers: List[int],
    model,
    preprocess,
    device: torch.device,
) -> Optional[Dict[int, np.ndarray]]:
    """Extract SigLIP image features for specific frames from a video.

    Args:
        video_path: Path to the video file.
        frame_numbers: List of frame numbers to extract.
        model: Loaded SigLIP model.
        preprocess: Preprocessing transform.
        device: Torch device.

    Returns:
        Dict mapping frame_number -> 1152-dim float32 feature vector,
        or None if the video cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video {video_path}")
        return None

    sorted_frames = sorted(frame_numbers)
    tensors = []
    for frame_num in sorted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Cannot read frame {frame_num} from {video_path}")
            cap.release()
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensors.append(preprocess(pil_image))

    cap.release()

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features = F.normalize(image_features, dim=-1)

    features_np = image_features.cpu().numpy().astype(np.float32)
    return {fn: features_np[i] for i, fn in enumerate(sorted_frames)}


def save_features_npz(
    features: Dict[int, np.ndarray],
    frame_numbers: List[int],
    output_path: str,
) -> None:
    """Save extracted features in the training data NPZ format.

    Args:
        features: Dict mapping frame_number -> 1152-dim feature vector.
        frame_numbers: Sorted list of 11 frame numbers.
        output_path: Path to save the .npz file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frame_nums_array = np.array(frame_numbers, dtype=np.int64)
    middle_frame = np.int64(frame_numbers[len(frame_numbers) // 2])

    feature_vectors = np.array([features[fn] for fn in frame_numbers], dtype=np.float32)
    averaged_features = feature_vectors.mean(axis=0)

    np.savez(
        output_path,
        features=np.array(features, dtype=object),
        frame_numbers=frame_nums_array,
        middle_frame=middle_frame,
        averaged_features=averaged_features,
        fish_species=np.str_(""),
    )


def classify_date_folder(
    date_folder: str,
    model,
    features_base_dir: str = "data/SigLIP_features",
) -> None:
    """Classify species for all NPZ files in a date folder.

    Args:
        date_folder: Date folder name (e.g., '06_08_2025').
        model: Loaded sklearn model with a predict() method.
        features_base_dir: Base directory for feature NPZ files.
    """
    features_dir = Path(features_base_dir) / date_folder
    if not features_dir.exists():
        print(f"Features directory not found: {features_dir}")
        return

    npz_files = sorted(features_dir.glob("*_features.npz"))
    if not npz_files:
        print(f"[{date_folder}] No NPZ files found")
        return

    classified = 0
    skipped = 0

    for npz_path in npz_files:
        data = dict(np.load(str(npz_path), allow_pickle=True))

        if str(data["fish_species"]) != "":
            skipped += 1
            continue

        SPECIES_LABELS = ["Bleikja", "Lax", "Urriði"]

        avg_features = data["averaged_features"].reshape(1, -1)
        prediction = model.predict(avg_features)[0]

        # Map numeric label to species name if needed
        if isinstance(prediction, (int, np.integer)):
            prediction = SPECIES_LABELS[prediction]

        data["fish_species"] = np.str_(prediction)
        np.savez(str(npz_path), **data)
        classified += 1

    print(f"[{date_folder}] Classification done: {classified} classified, {skipped} skipped")


def process_date_folder(
    date_folder: str,
    detection_dir: str = "output/detection_output",
    video_base_dir: str = "data",
    output_base_dir: str = "data/SigLIP_features",
) -> None:
    """Process all videos in a date folder and extract features.

    Args:
        date_folder: Date folder name (e.g., '06_08_2025').
        detection_dir: Base directory for detection output.
        video_base_dir: Base directory for source videos.
        output_base_dir: Base directory for feature output.
    """
    detection_path = Path(detection_dir) / date_folder / "fish_detection" / "results.json"
    multi_fish_path = Path(detection_dir) / date_folder / "multi_fish" / "results.json"

    if not detection_path.exists():
        print(f"Detection results not found: {detection_path}")
        return
    if not multi_fish_path.exists():
        print(f"Multi-fish results not found: {multi_fish_path}")
        return

    with open(detection_path) as f:
        detection_results = json.load(f)
    with open(multi_fish_path) as f:
        multi_fish_results = json.load(f)

    valid_videos = filter_videos(detection_results, multi_fish_results)
    print(f"[{date_folder}] {len(valid_videos)} videos with valid segments "
          f"(out of {len(detection_results)} total)")

    if not valid_videos:
        return

    print("Loading SigLIP model...")
    model, preprocess, device = load_siglip_model()
    print(f"Model loaded on {device}")

    output_dir = Path(output_base_dir) / date_folder
    processed = 0
    skipped = 0

    for video_path, segments in valid_videos.items():
        video_stem = Path(video_path).stem

        for seg_idx, seg_frames in enumerate(segments, start=1):
            output_path = output_dir / f"{video_stem}_seg{seg_idx}_features.npz"

            if output_path.exists():
                skipped += 1
                continue

            best_frames = select_best_window(seg_frames)
            if best_frames is None:
                continue

            video_file = str(Path(video_base_dir) / date_folder / Path(video_path).name)
            if not Path(video_file).exists():
                video_file = video_path
            if not Path(video_file).exists():
                print(f"  Video not found: {video_file}")
                continue

            features = extract_siglip_features(
                video_file, best_frames, model, preprocess, device
            )
            if features is None:
                continue

            save_features_npz(features, best_frames, str(output_path))
            processed += 1

    print(f"[{date_folder}] Done: {processed} extracted, {skipped} cached")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SigLIP features from fish video segments"
    )
    parser.add_argument("--date", type=str, help="Single date folder to process")
    parser.add_argument("--all", action="store_true", help="Process all date folders")
    parser.add_argument("--detection-dir", default="output/detection_output")
    parser.add_argument("--video-dir", default="data")
    parser.add_argument("--output-dir", default="data/SigLIP_features")
    parser.add_argument("--classify", action="store_true",
                        help="Run species classification on extracted features")
    parser.add_argument("--model-path",
                        default="notebooks/model_optimization_20260420_101150_multiseed/SVM_best_model.joblib",
                        help="Path to the SVM model joblib file")
    args = parser.parse_args()

    if args.classify:
        if not Path(args.model_path).exists():
            print(f"Model not found: {args.model_path}")
            return
        print(f"Loading model from {args.model_path}...")
        model = joblib.load(args.model_path)

        if args.date:
            classify_date_folder(args.date, model, args.output_dir)
        elif args.all:
            features_base = Path(args.output_dir)
            date_folders = sorted([
                d.name for d in features_base.iterdir() if d.is_dir()
            ])
            print(f"Found {len(date_folders)} date folders: {date_folders}")
            for date_folder in date_folders:
                classify_date_folder(date_folder, model, args.output_dir)
        else:
            print("Specify --date or --all with --classify")
    elif args.date:
        process_date_folder(args.date, args.detection_dir, args.video_dir, args.output_dir)
    elif args.all:
        detection_base = Path(args.detection_dir)
        date_folders = sorted([
            d.name for d in detection_base.iterdir()
            if d.is_dir() and (d / "fish_detection" / "results.json").exists()
        ])
        print(f"Found {len(date_folders)} date folders: {date_folders}")
        for date_folder in date_folders:
            process_date_folder(date_folder, args.detection_dir, args.video_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
