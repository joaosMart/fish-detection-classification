#!/usr/bin/env python3
"""
Fish Detection App
A simple application to detect fish in video files using AI.

Usage:
    python fish_detector.py single --video-dir ./videos
    python fish_detector.py multi --video-dir ./videos
    python fish_detector.py single --videos video1.mp4 video2.mp4
    python fish_detector.py multi --videos video1.mp4 video2.mp4
"""

import os
import sys
import json
import pickle
import time
import platform
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal

# Check and install requirements
def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'torch', 'torchvision', 'transformers', 
        'open_clip_torch', 'opencv-python', 
        'pillow', 'numpy', 'matplotlib', 'tqdm',
        'typing'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Installing missing packages...")
        import subprocess
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Installation complete!")

# Install requirements first
install_requirements()

# Now import the packages
import torch
import torch.nn.functional as F
import open_clip
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


class FishDetector:
    def __init__(self, detection_type: Literal["single", "multi"], output_dir: str = "./output/detection_output", 
                 video_dir: Optional[str] = None):
        self.base_output_dir = Path(output_dir)
        self.detection_type = detection_type
        
        # Create session-specific output directory based on video directory name
        if video_dir:
            video_dir_name = Path(video_dir).name
            self.output_dir = self.base_output_dir / video_dir_name
        else:
            self.output_dir = self.base_output_dir / "default_session"
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device with better cross-platform support
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device} on {platform.system()}")
        print(f"Detection mode: {self.detection_type}")
        print(f"Output directory: {self.output_dir}")
        
        if self.detection_type == "single":
            # File paths
            self.scores_file = self.output_dir / "fish_detection/scores.pkl"
            self.results_file = self.output_dir / "fish_detection/results.json"
            # Detection threshold
            self.threshold = 0.977989

        elif self.detection_type == "multi":
            # File paths
            self.scores_file = self.output_dir / "multi_fish/scores.pkl"
            self.results_file = self.output_dir / "multi_fish/results.json"
            # Path to single fish detection results (for optimization)
            self.single_results_file = self.output_dir / "fish_detection/results.json"
            # Multi Fish Detection Threshold 
            self.threshold = 0.9620
        
        # Create subdirectories
        self.scores_file.parent.mkdir(parents=True, exist_ok=True)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self.model = None
        self.preprocess = None
        self.text_features = None
        
        
        
    def setup_model(self):
        """Load and setup the AI model"""
        print("Loading AI model...")
        
        model_name = 'ViT-SO400M-14-SigLIP'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='webli'
        )
        self.model = self.model.to(self.device)
        tokenizer = open_clip.get_tokenizer(model_name)
        
        if self.detection_type == "single":
            # Setup text prompts for fish detection
            positive_prompts = tokenizer([
                "Salmon-like fish swimming",
                "An underwater photo of a salmon-like fish seen clearly swimming.",
                "Image of salmon-like fish in a contained environment.",
                "A photo of a salmon-like fish in a controlled river environment."
            ], context_length=self.model.context_length).to(self.device) #type: ignore
            
            negative_prompts = tokenizer([
                "An image of an empty white water container.",
                "A contained environment with nothing in it.",
                "An image of a empty container with nothing in it."
            ], context_length=self.model.context_length).to(self.device) #type: ignore

        elif self.detection_type == "multi":
            # Setup prompts for multi-fish detection
            positive_prompts = tokenizer([
                "Salmon-like fishes swimming",
                "Image of two or more salmon-like fish in a contained environment."
            ], context_length = self.model.context_length).to(self.device)  #type: ignore

            negative_prompts = tokenizer([
                "Clear image of a single fish swimming in a river."
            ], context_length=self.model.context_length).to(self.device) #type: ignore

        # Encode text features with cross-platform autocast
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    pos_features = self.model.encode_text(positive_prompts)  #type: ignore
                    neg_features = self.model.encode_text(negative_prompts)  #type: ignore
            elif self.device.type == 'mps':
                # Use MPS autocast for Mac Metal
                try:
                    if hasattr(torch, 'autocast') and hasattr(torch.autocast, '__call__'):
                        with torch.autocast(device_type='mps', dtype=torch.float16):
                            pos_features = self.model.encode_text(positive_prompts) #type: ignore
                            neg_features = self.model.encode_text(negative_prompts) #type: ignore
                    else: 
                        # Fallback for older PyTorch versions
                        pos_features = self.model.encode_text(positive_prompts) #type: ignore
                        neg_features = self.model.encode_text(negative_prompts) #type: ignore
                except Exception as e:
                    # Final fallback if MPS has issues
                    print(f"MPS autocast failed during text encoding ({e}), using standard processing")
                    pos_features = self.model.encode_text(positive_prompts) #type: ignore
                    neg_features = self.model.encode_text(negative_prompts) #type: ignore
            else:
                # CPU - no autocast needed
                pos_features = self.model.encode_text(positive_prompts) #type: ignore
                neg_features = self.model.encode_text(negative_prompts) #type: ignore
            
            self.text_features = torch.stack((neg_features.mean(axis=0), pos_features.mean(axis=0)))
            self.text_features = F.normalize(self.text_features, dim=-1)
        
        print("Model loaded successfully!")
    
    def process_batch_with_autocast(self, batch_tensor):
        """Process batch with appropriate autocast for the device"""
        with torch.no_grad():
            if self.device.type == 'cuda':
                # Use CUDA autocast for NVIDIA GPUs
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(batch_tensor) #type: ignore
                    image_features = F.normalize(image_features, dim=-1)
                    text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1) #type: ignore
            elif self.device.type == 'mps':
                # Use MPS for Mac Metal - handle different PyTorch versions
                try:
                    # Try newer PyTorch autocast syntax
                    if hasattr(torch, 'autocast') and hasattr(torch.autocast, '__call__'):
                        with torch.autocast(device_type='mps', dtype=torch.float16):
                            image_features = self.model.encode_image(batch_tensor) #type: ignore
                            image_features = F.normalize(image_features, dim=-1)
                            text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1) #type: ignore
                    else:
                        # Fallback for older PyTorch versions
                        image_features = self.model.encode_image(batch_tensor) #type: ignore
                        image_features = F.normalize(image_features, dim=-1)
                        text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1) #type: ignore
                except Exception as e:
                    # Final fallback if MPS has issues
                    print(f"MPS autocast failed ({e}), using standard processing")
                    image_features = self.model.encode_image(batch_tensor) #type: ignore
                    image_features = F.normalize(image_features, dim=-1)
                    text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1) #type: ignore
            else:
                # CPU - no autocast needed
                image_features = self.model.encode_image(batch_tensor) #type: ignore
                image_features = F.normalize(image_features, dim=-1)
                text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1) #type: ignore
        
        return text_probs
    
    def load_single_fish_results(self) -> Dict[str, List[int]]:
        """Load single fish detection results to get frames with fish"""
        if not self.single_results_file.exists():
            print(f"Warning: Single fish detection results not found at {self.single_results_file}")
            print("Please run single fish detection first!")
            return {}
        
        try:
            with open(self.single_results_file, 'r') as f:
                single_results = json.load(f)
            
            # Extract fish frame indices for each video
            fish_frames_by_video = {}
            for video_path, results in single_results.items():
                fish_frame_indices = [frame_info['frame'] for frame_info in results['fish_frames']]
                fish_frames_by_video[video_path] = fish_frame_indices
                
            print(f"Loaded single fish detection results for {len(fish_frames_by_video)} videos")
            return fish_frames_by_video
        except Exception as e:
            print(f"Error loading single fish results: {e}")
            return {}

    def process_video_optimized(self, video_path: str, fish_frame_indices: List[int]) -> List[Tuple[int, List[float]]]:
        """Process only specific frames from a video (optimized for multi-fish detection)"""
        if not fish_frame_indices:
            print(f"No fish frames found for {Path(video_path).name}, skipping...")
            return []
            
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            print(f"Warning: Could not read video {video_path}")
            cap.release()
            return []
        
        print(f"Processing {Path(video_path).name} - analyzing {len(fish_frame_indices)} fish frames out of {frame_count} total frames")
        
        results = []
        batch_size = 128
        batch = []
        batch_frame_indices = []
        
        # Sort frame indices for efficient video seeking
        sorted_fish_frames = sorted(fish_frame_indices)
        
        for target_frame_idx in tqdm(sorted_fish_frames, desc="Processing fish frames"):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {target_frame_idx} from {video_path}")
                continue
            
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.preprocess(image).unsqueeze(0) #type: ignore
            batch.append(image_tensor)
            batch_frame_indices.append(target_frame_idx)
            
            # Process batch when full
            if len(batch) == batch_size:
                batch_tensor = torch.cat(batch).to(self.device)
                text_probs = self.process_batch_with_autocast(batch_tensor)
                # Store results with original frame indices
                batch_results = text_probs.cpu().numpy().tolist()
                for i, probs in enumerate(batch_results):
                    results.append((batch_frame_indices[i], probs))
                batch = []
                batch_frame_indices = []
        
        # Process remaining frames
        if batch:
            batch_tensor = torch.cat(batch).to(self.device)
            text_probs = self.process_batch_with_autocast(batch_tensor)
            batch_results = text_probs.cpu().numpy().tolist()
            for i, probs in enumerate(batch_results):
                results.append((batch_frame_indices[i], probs))
        
        cap.release()
        return results

    def process_video(self, video_path: str) -> List[List[float]]:
        """Process a single video and return probability scores (every third frame)"""
        cap = cv2.VideoCapture(str(video_path))  # Ensure string path for cross-platform
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            print(f"Warning: Could not read video {video_path}")
            cap.release()
            return []
        
        results = []
        batch_size = 128  # Smaller batch size for stability
        batch = []
        batch_frame_indices = []  # Track which frames we're processing
        
        # Calculate frames to process (every third frame: 1, 4, 7, 10, ...)
        frames_to_process = list(range(0, frame_count, 3))  # 1-indexed pattern: 1, 4, 7, 10...
        
        print(f"Processing {Path(video_path).name} ({len(frames_to_process)} frames out of {frame_count} total - every 3rd frame)...")
        
        for frame_idx in tqdm(frames_to_process, desc="Processing frames"):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.preprocess(image).unsqueeze(0) #type: ignore
            batch.append(image_tensor)
            batch_frame_indices.append(frame_idx)
            
            # Process batch when full
            if len(batch) == batch_size:
                batch_tensor = torch.cat(batch).to(self.device)
                text_probs = self.process_batch_with_autocast(batch_tensor)
                results.extend(text_probs.cpu().numpy().tolist())
                batch = []
                batch_frame_indices = []
        
        # Process remaining frames
        if batch:
            batch_tensor = torch.cat(batch).to(self.device)
            text_probs = self.process_batch_with_autocast(batch_tensor)
            results.extend(text_probs.cpu().numpy().tolist())
        
        cap.release()
        
        # Create full results array with None for skipped frames
        full_results = [None] * frame_count
        for i, result in enumerate(results):
            actual_frame_idx = frames_to_process[i] if i < len(frames_to_process) else None
            if actual_frame_idx is not None:
                full_results[actual_frame_idx] = result
        
        return full_results # type: ignore
    
    def analyze_videos(self, video_files: List[str]):
        """Process all videos and generate scores"""
        if not self.model:
            self.setup_model()
        
        all_results = {}
        
        if self.detection_type == "multi":
            # For multi-fish detection, load single fish results first
            fish_frames_by_video = self.load_single_fish_results()
            
            if not fish_frames_by_video:
                print("Cannot proceed with multi-fish detection without single fish results.")
                return {}
            
            for video_path in video_files:
                video_path_obj = Path(video_path)
                if not video_path_obj.exists():
                    print(f"Warning: Video not found: {video_path}")
                    continue
                
                # Get fish frames for this video
                fish_frames = fish_frames_by_video.get(str(video_path_obj), [])
                if not fish_frames:
                    print(f"No fish frames found for {video_path_obj.name} in single detection results")
                    continue
                
                # Process only fish frames
                frame_results = self.process_video_optimized(str(video_path_obj), fish_frames)
                if frame_results:
                    all_results[str(video_path_obj)] = frame_results
        
        else:  # Single fish detection
            for video_path in video_files:
                video_path_obj = Path(video_path)
                if not video_path_obj.exists():
                    print(f"Warning: Video not found: {video_path}")
                    continue
                    
                scores = self.process_video(str(video_path_obj))
                if scores:
                    all_results[str(video_path_obj)] = scores
        
        # Save raw scores
        with open(self.scores_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"Scores saved to: {self.scores_file}")
        return all_results
    
    def filter_results(self, scores: Dict[str, Any]):
        """Filter frames above threshold and generate final results (handles None values)"""
        filtered_results = {}
        
        if self.detection_type == "multi":
            # Handle multi-fish results (frame_idx, [no_fish_prob, fish_prob])
            for video_path, frame_data in scores.items():
                fish_frames = []
                
                for frame_idx, (no_fish_prob, fish_prob) in frame_data:
                    if fish_prob >= self.threshold:
                        fish_frames.append({
                            "frame": frame_idx,
                            "probability": fish_prob
                        })
                
                filtered_results[video_path] = {
                    "total_frames_analyzed": len(frame_data),
                    "multi_fish_frames": fish_frames
                }
        
        else:  # Single fish detection
            for video_path, frame_scores in scores.items():
                fish_frames = []
                frames_processed = 0
                
                for frame_idx, scores_data in enumerate(frame_scores):
                    if scores_data is not None:  # Only process non-None frames
                        frames_processed += 1
                        no_fish_prob, fish_prob = scores_data
                        if fish_prob >= self.threshold:
                            fish_frames.append({
                                "frame": frame_idx,
                                "probability": fish_prob
                            })
                
                filtered_results[video_path] = {
                    "total_frames": len(frame_scores),
                    "frames_processed": frames_processed,
                    "fish_frames": fish_frames
                }
        
        # Save filtered results
        with open(self.results_file, 'w') as f:
            json.dump(filtered_results, f, indent=2)
        
        print(f"Results saved to: {self.results_file}")
        return filtered_results
    
    def print_summary(self, results: Dict[str, Dict]):
        """Print a summary of detection results (updated for third-frame processing)"""
        total_videos = len(results)
        
        if self.detection_type == "multi":
            videos_with_multi_fish = sum(1 for r in results.values() if r.get('multi_fish_frames'))
            total_multi_fish_frames = sum(len(r.get('multi_fish_frames', [])) for r in results.values())
            total_analyzed_frames = sum(r.get('total_frames_analyzed', 0) for r in results.values())
            
            print("\n" + "="*50)
            print("MULTI-FISH DETECTION SUMMARY")
            print("="*50)
            print(f"Detection mode: {self.detection_type.upper()}")
            print(f"Frame processing: Every 3rd frame")
            print(f"Total videos processed: {total_videos}")
            print(f"Videos with multi-fish detected: {videos_with_multi_fish}")
            print(f"Videos without multi-fish: {total_videos - videos_with_multi_fish}")
            print(f"Total fish frames analyzed: {total_analyzed_frames}")
            print(f"Frames with multi-fish detected: {total_multi_fish_frames}")
            if total_analyzed_frames > 0:
                print(f"Multi-fish detection rate: {(total_multi_fish_frames/total_analyzed_frames*100):.2f}%")
            print(f"Detection threshold used: {self.threshold}")
        else:
            videos_with_fish = sum(1 for r in results.values() if r['fish_frames'])
            total_fish_frames = sum(len(r['fish_frames']) for r in results.values())
            total_frames = sum(r['total_frames'] for r in results.values())
            frames_processed = sum(r.get('frames_processed', 0) for r in results.values())
            
            print("\n" + "="*50)
            print("SINGLE FISH DETECTION SUMMARY")
            print("="*50)
            print(f"Detection mode: {self.detection_type.upper()}")
            print(f"Frame processing: Every 3rd frame")
            print(f"Total videos processed: {total_videos}")
            print(f"Videos with fish detected: {videos_with_fish}")
            print(f"Videos without fish: {total_videos - videos_with_fish}")
            print(f"Total frames in videos: {total_frames}")
            print(f"Frames actually processed: {frames_processed}")
            print(f"Processing efficiency: {(frames_processed/total_frames*100):.1f}% of frames")
            print(f"Frames with fish detected: {total_fish_frames}")
            print(f"Detection rate: {(total_fish_frames/frames_processed*100):.2f}%")
            print(f"Detection threshold used: {self.threshold}")
        
        print("\nDetailed results saved to:")
        print(f"  - Raw scores: {self.scores_file}")
        print(f"  - Filtered results: {self.results_file}")
        print(f"  - Session output directory: {self.output_dir}")
    
    def identify_segments(self, filtered_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Identify fish segments in filtered results.

        Groups consecutive fish detections into segments, fills intermediate
        frames, and enriches results with segment assignments.

        Thresholds are derived from each video's fps:
        - gap_threshold: 2 * fps (2 seconds gap = new segment)
        - min_segment_length: fps (1 second of fish = valid segment)
        """
        from segment_utils import find_segments

        for video_path, data in filtered_results.items():
            fish_frames = data.get('fish_frames', [])

            if not fish_frames:
                data['segments_summary'] = {'total_segments': 0, 'segments': []}
                continue

            # Read fps from video to compute dynamic thresholds
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0:
                fps = 10  # fallback

            gap_threshold = int(2 * fps)
            min_segment_length = int(fps)

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

    def run(self, video_files: List[str]):
        """Run the complete fish detection pipeline"""
        print(f"Starting {self.detection_type.upper()} Fish Detection...")
        print(f"Videos to process: {len(video_files)}")

        # Process videos
        scores = self.analyze_videos(video_files)

        if not scores:
            print("No videos were successfully processed!")
            return

        # Filter results
        results = self.filter_results(scores)

        # Identify segments (single fish detection only)
        if self.detection_type == "single":
            results = self.identify_segments(results)

        # Print summary
        self.print_summary(results)

def get_video_files_from_directory(directory_path: Path) -> List[str]:
    """Get video files from a directory"""
    if not directory_path.exists():
        print(f"Error: Directory not found: {directory_path}")
        return []
    
    # Find video files in directory
    video_extensions = ['.mp4']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(directory_path.glob(f"*{ext}"))
        video_files.extend(directory_path.glob(f"*{ext.upper()}"))
    
    video_files = [str(f) for f in video_files]
    print(f"Found {len(video_files)} video files in {directory_path}")
    return video_files

def validate_video_files(video_paths: List[str]) -> List[str]:
    """Validate that video files exist"""
    valid_files = []
    for video_path in video_paths:
        path_obj = Path(video_path)
        if path_obj.exists():
            valid_files.append(str(path_obj))
        else:
            print(f"Warning: Video file not found: {video_path}")
    
    print(f"Processing {len(valid_files)} valid video files")
    return valid_files

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='🐟 Fish Detection App - Detect fish in video files using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single --video-dir ./videos
  %(prog)s multi --video-dir ./videos  
  %(prog)s single --videos video1.mp4 video2.mp4
  %(prog)s multi --videos video1.mp4 video2.mp4
  %(prog)s single --video-dir ./data --output-dir ./results
        """
    )
    
    # Make mode required
    parser.add_argument('mode', choices=['single', 'multi'], 
                       help='Detection mode: single (individual fish) or multi (multiple fish)')
    
    parser.add_argument('--output-dir', default='./output/detection_output',
                       help='Output directory for results (default: ./output/detection_output)')
    
    # Video input options - make them mutually exclusive
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument('--video-dir', 
                           help='Directory containing video files to process')
    video_group.add_argument('--videos', nargs='+',
                           help='Specific video files to process')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    args = parse_arguments()
    
    print("🐟 Fish Detection App 🐟")
    print(f"Mode: {args.mode.upper()} fish detection")
    print("-" * 50)
    
    # Get video files and determine video directory
    video_files = []
    video_dir = None
    
    if args.videos:
        # Videos specified via command line
        video_files = validate_video_files(args.videos)
        # Use parent directory of first video as session identifier
        if video_files:
            video_dir = str(Path(video_files[0]).parent)
    elif args.video_dir:
        # Video directory specified via command line
        directory_path = Path(args.video_dir)
        video_files = get_video_files_from_directory(directory_path)
        video_dir = str(directory_path)
    
    if not video_files:
        print("Error: No valid video files found. Exiting.")
        sys.exit(1)
    
    # Create detector and run
    detector = FishDetector(args.mode, args.output_dir, video_dir)
    detector.run(video_files)
    
    print(f"\n🎉 {args.mode.upper()} fish detection complete!")
    print(f"Check the session output directory for results: {detector.output_dir}")

if __name__ == "__main__":
    main()