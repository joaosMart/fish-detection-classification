#!/usr/bin/env python3
"""
Fish Detection Timeline Viewer
Generates viewer_data.json from detection results, copies videos,
and serves an interactive web viewer.
"""

import os
import sys
import json
import shutil
import argparse
import webbrowser
import threading
import time
import socket
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import cv2
from http.server import HTTPServer, SimpleHTTPRequestHandler


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video FPS and duration."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return {"fps": fps if fps > 0 else 30.0, "duration": duration}
    except Exception:
        return {"fps": 30.0, "duration": 0}


def load_json(path: Path) -> Dict:
    """Load a JSON file, return empty dict on failure."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def generate_viewer_data(session_dir: Path) -> Optional[Dict]:
    """
    Read single and multi detection results, merge into unified viewer_data.json format.
    Returns the data dict, or None if no results found.
    """
    single_results = load_json(session_dir / "fish_detection" / "results.json")
    multi_results = load_json(session_dir / "multi_fish" / "results.json")

    if not single_results and not multi_results:
        print("No detection results found.")
        return None

    # Collect all video paths from both result sets
    all_video_paths = set(single_results.keys()) | set(multi_results.keys())

    videos = {}
    for video_path_str in all_video_paths:
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Warning: video not found: {video_path_str}")
            continue

        video_info = get_video_info(str(video_path))
        fps = video_info["fps"]
        duration = video_info["duration"]
        video_name = video_path.name

        # Build single detections
        single_detections = []
        single_data = single_results.get(video_path_str, {})
        for det in single_data.get("fish_frames", []):
            frame = det["frame"]
            single_detections.append({
                "frame": frame,
                "time": round(frame / fps, 2),
                "probability": round(det.get("probability") or 0.0, 4),
            })

        # Build multi detections
        multi_detections = []
        multi_data = multi_results.get(video_path_str, {})
        for det in multi_data.get("multi_fish_frames", []):
            frame = det["frame"]
            multi_detections.append({
                "frame": frame,
                "time": round(frame / fps, 2),
                "probability": round(det.get("probability") or 0.0, 4),
            })

        # Build segments from single detection data
        segments = []
        segments_summary = single_data.get("segments_summary", {})
        if segments_summary.get("segments"):
            for seg in segments_summary["segments"]:
                # Look up species from NPZ features file
                video_stem = video_path.stem
                seg_num = seg["segment_number"]
                npz_path = Path("data/SigLIP_features") / session_dir.name / f"{video_stem}_seg{seg_num}_features.npz"
                species = "No Species Predicted"
                if npz_path.exists():
                    try:
                        npz_data = np.load(str(npz_path), allow_pickle=True)
                        sp = str(npz_data["fish_species"])
                        if sp:
                            species = sp
                    except Exception:
                        pass

                segments.append({
                    "segment_number": seg["segment_number"],
                    "start_time": round(seg["start_frame"] / fps, 2),
                    "end_time": round(seg["end_frame"] / fps, 2),
                    "start_frame": seg["start_frame"],
                    "end_frame": seg["end_frame"],
                    "size": seg["size"],
                    "species": species,
                })

        videos[video_name] = {
            "duration": round(duration, 2),
            "fps": round(fps, 1),
            "single_detections": single_detections,
            "multi_detections": multi_detections,
            "segments": segments,
            "source_path": video_path_str,
        }

    if not videos:
        print("No videos could be processed.")
        return None

    return {
        "session": session_dir.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "videos": videos,
    }


def build_output(session_dir: Path, viewer_data: Dict) -> Path:
    """
    Prepare viewer app files and viewer_data.json in the output directory.
    Videos are served directly from their original locations (no copying).
    Returns the output directory path.
    """
    output_dir = session_dir / "viewer_app"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy viewer app files
    viewer_src = Path(__file__).parent / "viewer"
    for fname in ["index.html", "style.css", "app.js"]:
        src = viewer_src / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
        else:
            print(f"Warning: viewer file not found: {src}")

    # Write viewer_data.json
    with open(output_dir / "viewer_data.json", "w") as f:
        json.dump(viewer_data, f, indent=2)

    return output_dir


def find_available_port(start: int = 8000) -> int:
    """Find an available port."""
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    return start


def start_server(directory: Path, port: int):
    """Start HTTP server serving from directory, with video proxying from original paths."""
    project_root = Path(__file__).parent.resolve()
    os.chdir(directory)

    class ViewerHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/serve-video?path="):
                from urllib.parse import unquote, urlparse, parse_qs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                video_path = unquote(params.get("path", [""])[0])
                # Resolve relative paths against the project root
                if video_path and not os.path.isabs(video_path):
                    video_path = str(project_root / video_path)
                if not video_path or not os.path.isfile(video_path):
                    self.send_error(404, "Video not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(os.path.getsize(video_path)))
                self.end_headers()
                with open(video_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            super().do_GET()

    server = HTTPServer(("localhost", port), ViewerHandler)
    server.serve_forever()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="🐟 Fish Detection Timeline Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --session 14_08_2025
  %(prog)s --session 14_08_2025 --no-browser
        """,
    )
    parser.add_argument(
        "--output-dir",
        default="./output/detection_output",
        help="Base output directory (default: ./output/detection_output)",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session name (directory containing detection results)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("🐟 Fish Detection Timeline Viewer")
    print(f"Session: {args.session}")
    print("-" * 50)

    session_dir = Path(args.output_dir) / args.session
    if not session_dir.exists():
        print(f"Error: session directory not found: {session_dir}")
        available = [d.name for d in Path(args.output_dir).glob("*") if d.is_dir()]
        if available:
            print("Available sessions:", ", ".join(available))
        sys.exit(1)

    # Generate unified data
    print("Generating viewer data...")
    viewer_data = generate_viewer_data(session_dir)
    if not viewer_data:
        sys.exit(1)

    total_single = sum(
        len(v["single_detections"]) for v in viewer_data["videos"].values()
    )
    total_multi = sum(
        len(v["multi_detections"]) for v in viewer_data["videos"].values()
    )
    print(
        f"Found {len(viewer_data['videos'])} videos, "
        f"{total_single} single detections, {total_multi} multi detections"
    )

    # Build output directory
    print("\nBuilding viewer app...")
    output_dir = build_output(session_dir, viewer_data)
    print(f"Viewer app ready at: {output_dir}")

    # Serve
    if not args.no_browser:
        port = find_available_port()
        server_thread = threading.Thread(
            target=start_server, args=(output_dir, port), daemon=True
        )
        server_thread.start()
        time.sleep(0.5)

        url = f"http://localhost:{port}"
        print(f"\n{'=' * 50}")
        print(f"🚀 Viewer running at: {url}")
        print(f"{'=' * 50}")
        print("Press Ctrl+C to stop the server")
        webbrowser.open(url)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")
    else:
        print(f"\nOpen {output_dir / 'index.html'} with a local server to view.")


if __name__ == "__main__":
    main()
