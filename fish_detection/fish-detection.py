#!/usr/bin/env python3
"""
Fish Detection App
A simple application to detect fish in video files using AI.

Usage:
    python fish_detector.py
"""

import os
import sys
import json
import pickle
import time
import platform
from pathlib import Path
from typing import List, Dict, Any

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
from typing import Literal

class FishDetector:
    def __init__(self, detection_type: Literal["single", "multi"], output_dir: str = "./data/detection_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.detection_type = detection_type
        
        # Detect device with better cross-platform support
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device} on {platform.system()}")
        
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
            # Multi Fish Detection Threshold 
            self.threshold = 0.9620
        
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
        
        # Setup text prompts for fish detection
        positive_prompts = tokenizer([
            "Salmon-like fish swimming",
            "An underwater photo of a salmon-like fish seen clearly swimming.",
            "Image of salmon-like fish in a contained environment.",
            "A photo of a salmon-like fish in a controlled river environment."
        ], context_length=self.model.context_length).to(self.device)
        
        negative_prompts = tokenizer([
            "An image of an empty white water container.",
            "A contained environment with nothing in it.",
            "An image of a empty container with nothing in it."
        ], context_length=self.model.context_length).to(self.device)
        
        # Encode text features with cross-platform autocast
        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    pos_features = self.model.encode_text(positive_prompts) 
                    neg_features = self.model.encode_text(negative_prompts)
            elif self.device.type == 'mps':
                # Use MPS autocast for Mac Metal
                try:
                    if hasattr(torch, 'autocast') and hasattr(torch.autocast, '__call__'):
                        with torch.autocast(device_type='mps', dtype=torch.float16):
                            pos_features = self.model.encode_text(positive_prompts)
                            neg_features = self.model.encode_text(negative_prompts)
                    else:
                        # Fallback for older PyTorch versions
                        pos_features = self.model.encode_text(positive_prompts)
                        neg_features = self.model.encode_text(negative_prompts)
                except Exception as e:
                    # Final fallback if MPS has issues
                    print(f"MPS autocast failed during text encoding ({e}), using standard processing")
                    pos_features = self.model.encode_text(positive_prompts)
                    neg_features = self.model.encode_text(negative_prompts)
            else:
                # CPU - no autocast needed
                pos_features = self.model.encode_text(positive_prompts)
                neg_features = self.model.encode_text(negative_prompts)
            
            self.text_features = torch.stack((neg_features.mean(axis=0), pos_features.mean(axis=0)))
            self.text_features = F.normalize(self.text_features, dim=-1)
        
        print("Model loaded successfully!")
    
    def process_batch_with_autocast(self, batch_tensor):
        """Process batch with appropriate autocast for the device"""
        with torch.no_grad():
            if self.device.type == 'cuda':
                # Use CUDA autocast for NVIDIA GPUs
                with torch.cuda.amp.autocast():
                    image_features = self.model.encode_image(batch_tensor)
                    image_features = F.normalize(image_features, dim=-1)
                    text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            elif self.device.type == 'mps':
                # Use MPS for Mac Metal - handle different PyTorch versions
                try:
                    # Try newer PyTorch autocast syntax
                    if hasattr(torch, 'autocast') and hasattr(torch.autocast, '__call__'):
                        with torch.autocast(device_type='mps', dtype=torch.float16):
                            image_features = self.model.encode_image(batch_tensor)
                            image_features = F.normalize(image_features, dim=-1)
                            text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                    else:
                        # Fallback for older PyTorch versions
                        image_features = self.model.encode_image(batch_tensor)
                        image_features = F.normalize(image_features, dim=-1)
                        text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                except Exception as e:
                    # Final fallback if MPS has issues
                    print(f"MPS autocast failed ({e}), using standard processing")
                    image_features = self.model.encode_image(batch_tensor)
                    image_features = F.normalize(image_features, dim=-1)
                    text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            else:
                # CPU - no autocast needed
                image_features = self.model.encode_image(batch_tensor)
                image_features = F.normalize(image_features, dim=-1)
                text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        
        return text_probs
    
    def process_video(self, video_path: str) -> List[List[float]]:
        """Process a single video and return probability scores"""
        cap = cv2.VideoCapture(str(video_path))  # Ensure string path for cross-platform
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            print(f"Warning: Could not read video {video_path}")
            cap.release()
            return []
        
        results = []
        batch_size = 32  # Smaller batch size for stability
        batch = []
        
        print(f"Processing {Path(video_path).name} ({frame_count} frames)...")
        
        for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.preprocess(image).unsqueeze(0)
            batch.append(image_tensor)
            
            # Process batch when full
            if len(batch) == batch_size:
                batch_tensor = torch.cat(batch).to(self.device)
                text_probs = self.process_batch_with_autocast(batch_tensor)
                results.extend(text_probs.cpu().numpy().tolist())
                batch = []
        
        # Process remaining frames
        if batch:
            batch_tensor = torch.cat(batch).to(self.device)
            text_probs = self.process_batch_with_autocast(batch_tensor)
            results.extend(text_probs.cpu().numpy().tolist())
        
        cap.release()
        return results
    
    def analyze_videos(self, video_files: List[str]):
        """Process all videos and generate scores"""
        if not self.model:
            self.setup_model()
        
        all_results = {}
        
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
    
    def filter_results(self, scores: Dict[str, List[List[float]]]):
        """Filter frames above threshold and generate final results"""
        filtered_results = {}
        
        for video_path, frame_scores in scores.items():
            fish_frames = []
            
            for frame_idx, (no_fish_prob, fish_prob) in enumerate(frame_scores):
                if fish_prob >= self.threshold:
                    fish_frames.append({
                        "frame": frame_idx,
                        "probability": fish_prob
                    })
            
            filtered_results[video_path] = {
                "total_frames": len(frame_scores),
                "fish_frames": fish_frames
            }
        
        # Save filtered results
        with open(self.results_file, 'w') as f:
            json.dump(filtered_results, f, indent=2)
        
        print(f"Results saved to: {self.results_file}")
        return filtered_results
    
    def print_summary(self, results: Dict[str, Dict]):
        """Print a summary of detection results"""
        total_videos = len(results)
        videos_with_fish = sum(1 for r in results.values() if r['fish_frames'])
        total_fish_frames = sum(len(r['fish_frames']) for r in results.values())
        total_frames = sum(r['total_frames'] for r in results.values())
        
        print("\n" + "="*50)
        print("FISH DETECTION SUMMARY")
        print("="*50)
        print(f"Total videos processed: {total_videos}")
        print(f"Videos with fish detected: {videos_with_fish}")
        print(f"Videos without fish: {total_videos - videos_with_fish}")
        print(f"Total frames analyzed: {total_frames}")
        print(f"Frames with fish detected: {total_fish_frames}")
        print(f"Detection rate: {(total_fish_frames/total_frames*100):.2f}%")
        print(f"Detection threshold used: {self.threshold}")
        print("\nDetailed results saved to:")
        print(f"  - Raw scores: {self.scores_file}")
        print(f"  - Filtered results: {self.results_file}")
    
    def run(self, video_files: List[str]):
        """Run the complete fish detection pipeline"""
        print("Starting Fish Detection Pipeline...")
        print(f"Videos to process: {len(video_files)}")
        
        # Process videos
        scores = self.analyze_videos(video_files)
        
        if not scores:
            print("No videos were successfully processed!")
            return
        
        # Filter results
        results = self.filter_results(scores)
        
        # Print summary
        self.print_summary(results)

def get_video_files():
    """Get video files from user input"""
    print("\nHow would you like to specify video files?")
    print("1. Enter video file paths manually")
    print("2. Specify a directory containing videos")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        video_files = []
        print("\nEnter video file paths (one per line, press Enter twice when done):")
        while True:
            path = input().strip()
            if not path:
                break
            path_obj = Path(path)
            if path_obj.exists():
                video_files.append(str(path_obj))
            else:
                print(f"Warning: File not found: {path}")
        return video_files
    
    elif choice == "2":
        directory = input("Enter directory path: ").strip()
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"Directory not found: {directory}")
            return []
        
        # Find video files in directory
        video_extensions = ['.mp4']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(directory_path.glob(f"*{ext}"))
            video_files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        video_files = [str(f) for f in video_files]
        print(f"Found {len(video_files)} video files")
        return video_files
    
    else:
        print("Invalid choice!")
        return []

def main():
    """Main application entry point"""
    print("🐟 Fish Detection App 🐟")
    print("This app uses AI to detect fish in video files.")
    print("-" * 50)
    
    # Get video files
    video_files = get_video_files()
    
    if not video_files:
        print("No video files specified. Exiting.")
        return
    
    # Ask for output directory
    output_dir = input(f"\nOutput directory (press Enter for './data/detection_output'): ").strip()
    if not output_dir:
        output_dir = "./data/fish_detection_output"
    
    # Create detector and run
    detector = FishDetector(output_dir)
    detector.run(video_files)
    
    print("\n🎉 Detection complete! Check the output directory for results.")

if __name__ == "__main__":
    main()