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


def _find_multi_fish_segments(multi_fish_frames: List[Dict], min_run: int = 11) -> List[set]:
    """Find runs of consecutive multi-fish frames >= min_run length.

    Returns a list of sets, each containing frame numbers in a multi-fish run.
    """
    if not multi_fish_frames:
        return []

    frames = sorted(f["frame"] for f in multi_fish_frames)
    runs = []
    current_run = [frames[0]]

    for i in range(1, len(frames)):
        if frames[i] <= frames[i - 1] + 1:
            current_run.append(frames[i])
        else:
            if len(current_run) >= min_run:
                runs.append(set(current_run))
            current_run = [frames[i]]

    if len(current_run) >= min_run:
        runs.append(set(current_run))

    return runs


def filter_videos(
    detection_results: Dict[str, Dict],
    multi_fish_results: Dict[str, Dict],
    min_segment_size: int = 11,
) -> Dict[str, List[List[Dict]]]:
    """Filter videos to those with valid single-fish segments.

    Excludes segments that overlap with multi-fish runs of >= 11 consecutive
    frames. Also excludes segments shorter than min_segment_size.

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
        multi = multi_fish_results.get(video_path, {})
        multi_runs = _find_multi_fish_segments(
            multi.get("multi_fish_frames", []), min_run=min_segment_size
        )
        multi_frames = set()
        for run in multi_runs:
            multi_frames |= run

        # Group fish_frames by segment number
        segments_by_num: Dict[int, List[Dict]] = {}
        for frame in det["fish_frames"]:
            seg_num = frame.get("segment")
            if seg_num is None:
                continue
            segments_by_num.setdefault(seg_num, []).append(frame)

        # Filter segments by minimum size and multi-fish overlap
        valid_segments = []
        for seg_num in sorted(segments_by_num.keys()):
            seg_frames = sorted(segments_by_num[seg_num], key=lambda f: f["frame"])
            if len(seg_frames) < min_segment_size:
                continue
            seg_frame_nums = {f["frame"] for f in seg_frames}
            if seg_frame_nums & multi_frames:
                continue
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


SPECIES_LABELS = ["Bleikja", "Lax", "Urriði"]


def _read_npz_species(npz_path: Path) -> str:
    data = np.load(str(npz_path), allow_pickle=True)
    return str(data["fish_species"])


def _read_npz_middle_frame(npz_path: Path) -> Optional[int]:
    data = np.load(str(npz_path), allow_pickle=True)
    return int(data["middle_frame"]) if "middle_frame" in data.files else None


def write_species_results_json(
    session: str,
    detection_dir: str = "output/detection_output",
    features_base_dir: str = "data/SigLIP_features",
    output_base_dir: str = "output/species_classification",
) -> Optional[Path]:
    """Emit a per-segment species classification JSON for a session.

    Joins detection segments with NPZ species labels. Each segment gets a
    status: classified, too_short, multi_fish, or not_extracted.
    """
    detection_path = Path(detection_dir) / session / "fish_detection" / "results.json"
    multi_fish_path = Path(detection_dir) / session / "multi_fish" / "results.json"
    if not detection_path.exists():
        print(f"Detection results not found: {detection_path}")
        return None

    with open(detection_path) as f:
        detection_results = json.load(f)
    multi_fish_results = {}
    if multi_fish_path.exists():
        with open(multi_fish_path) as f:
            multi_fish_results = json.load(f)

    features_dir = Path(features_base_dir) / session
    min_segment_size = 11

    out: Dict[str, Dict] = {}
    for video_path, det in detection_results.items():
        video_stem = Path(video_path).stem
        segments = det.get("segments_summary", {}).get("segments", [])

        multi = multi_fish_results.get(video_path, {})
        multi_runs = _find_multi_fish_segments(
            multi.get("multi_fish_frames", []), min_run=min_segment_size
        )
        multi_frames: set = set()
        for run in multi_runs:
            multi_frames |= run

        seg_frames_by_num: Dict[int, set] = {}
        for fr in det.get("fish_frames", []):
            seg_num = fr.get("segment")
            if seg_num is None:
                continue
            seg_frames_by_num.setdefault(seg_num, set()).add(fr["frame"])

        out_segments = []
        for seg in segments:
            seg_num = seg["segment_number"]
            size = seg.get("size", 0)
            frames = seg_frames_by_num.get(seg_num, set())
            overlaps_multi = bool(frames & multi_frames)

            species = None
            middle_frame = None
            status = "classified"

            if size < min_segment_size:
                status = "too_short"
            elif overlaps_multi:
                status = "multi_fish"
            else:
                npz_path = features_dir / f"{video_stem}_seg{seg_num}_features.npz"
                if not npz_path.exists():
                    status = "not_extracted"
                else:
                    label = _read_npz_species(npz_path)
                    middle_frame = _read_npz_middle_frame(npz_path)
                    if label == "":
                        status = "not_classified"
                    else:
                        species = label

            out_segments.append({
                "segment_number": seg_num,
                "start_frame": seg.get("start_frame"),
                "end_frame": seg.get("end_frame"),
                "size": size,
                "middle_frame": middle_frame,
                "species": species,
                "status": status,
            })

        out[video_path] = {"segments": out_segments}

    output_path = Path(output_base_dir) / session / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[{session}] Species results written: {output_path}")
    return output_path


def classify_session(
    session: str,
    model,
    features_base_dir: str = "data/SigLIP_features",
    detection_dir: str = "output/detection_output",
    species_output_dir: str = "output/species_classification",
) -> None:
    """Classify species for all NPZ files in a session and emit results JSON.

    Args:
        session: Session folder name (typically a date like '06_08_2025').
        model: Loaded sklearn model with a predict() method.
        features_base_dir: Base directory for feature NPZ files.
        detection_dir: Base directory for detection output (for JSON export).
        species_output_dir: Where the per-segment species results.json is written.
    """
    features_dir = Path(features_base_dir) / session
    if not features_dir.exists():
        print(f"Features directory not found: {features_dir}")
        return

    npz_files = sorted(features_dir.glob("*_features.npz"))
    if not npz_files:
        print(f"[{session}] No NPZ files found")
        return

    classified = 0
    skipped = 0

    for npz_path in npz_files:
        data = dict(np.load(str(npz_path), allow_pickle=True))

        if str(data["fish_species"]) != "":
            skipped += 1
            continue

        avg_features = data["averaged_features"].reshape(1, -1)
        prediction = model.predict(avg_features)[0]

        if isinstance(prediction, (int, np.integer)):
            prediction = SPECIES_LABELS[prediction]

        data["fish_species"] = np.str_(prediction)
        np.savez(str(npz_path), **data)
        classified += 1

    print(f"[{session}] Classification done: {classified} classified, {skipped} skipped")

    write_species_results_json(
        session,
        detection_dir=detection_dir,
        features_base_dir=features_base_dir,
        output_base_dir=species_output_dir,
    )


# Backwards-compatible alias
classify_date_folder = classify_session


def process_session(
    session: str,
    detection_dir: str = "output/detection_output",
    video_base_dir: str = "data",
    output_base_dir: str = "data/SigLIP_features",
) -> None:
    """Process all videos in a session and extract features.

    Args:
        session: Session folder name (typically a date like '06_08_2025').
        detection_dir: Base directory for detection output.
        video_base_dir: Base directory for source videos.
        output_base_dir: Base directory for feature output.
    """
    detection_path = Path(detection_dir) / session / "fish_detection" / "results.json"
    multi_fish_path = Path(detection_dir) / session / "multi_fish" / "results.json"

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
    print(f"[{session}] {len(valid_videos)} videos with valid segments "
          f"(out of {len(detection_results)} total)")

    if not valid_videos:
        return

    print("Loading SigLIP model...")
    model, preprocess, device = load_siglip_model()
    print(f"Model loaded on {device}")

    output_dir = Path(output_base_dir) / session
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

            video_file = str(Path(video_base_dir) / session / Path(video_path).name)
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

    print(f"[{session}] Done: {processed} extracted, {skipped} cached")


# Backwards-compatible alias
process_date_folder = process_session


def main():
    parser = argparse.ArgumentParser(
        description="Extract SigLIP features from fish video segments"
    )
    parser.add_argument("--session", "--date", dest="session", type=str,
                        help="Single session folder to process (typically a date)")
    parser.add_argument("--all", action="store_true", help="Process all session folders")
    parser.add_argument("--detection-dir", default="output/detection_output")
    parser.add_argument("--video-dir", default="data")
    parser.add_argument("--output-dir", default="data/SigLIP_features")
    parser.add_argument("--species-output-dir", default="output/species_classification",
                        help="Where per-session species results.json is written")
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

        if args.session:
            classify_session(args.session, model, args.output_dir,
                             detection_dir=args.detection_dir,
                             species_output_dir=args.species_output_dir)
        elif args.all:
            features_base = Path(args.output_dir)
            sessions = sorted([d.name for d in features_base.iterdir() if d.is_dir()])
            print(f"Found {len(sessions)} sessions: {sessions}")
            for session in sessions:
                classify_session(session, model, args.output_dir,
                                 detection_dir=args.detection_dir,
                                 species_output_dir=args.species_output_dir)
        else:
            print("Specify --session or --all with --classify")
    elif args.session:
        process_session(args.session, args.detection_dir, args.video_dir, args.output_dir)
    elif args.all:
        detection_base = Path(args.detection_dir)
        sessions = sorted([
            d.name for d in detection_base.iterdir()
            if d.is_dir() and (d / "fish_detection" / "results.json").exists()
        ])
        print(f"Found {len(sessions)} sessions: {sessions}")
        for session in sessions:
            process_session(session, args.detection_dir, args.video_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
