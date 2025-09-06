#!/usr/bin/env python3
"""
Simple Fish Detection Scores Plotter

Usage:
    python simple_plot.py /path/to/your/video.mp4
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_fish_scores(video_path, output_dir="./data/fish_detection_output"):
    """Plot fish detection scores for a specific video"""
    
    # Load the scores
    scores_file = Path(output_dir) / "fish_scores.pkl"
    
    if not scores_file.exists():
        print(f"Error: Scores file not found: {scores_file}")
        print("Make sure you've run the fish detection first!")
        return
    
    with open(scores_file, 'rb') as f:
        scores = pickle.load(f)
    
    # Find the video (try exact path first, then just filename)
    video_path = str(video_path)
    if video_path not in scores:
        # Try to find by filename
        video_name = Path(video_path).name
        found_path = None
        for path in scores.keys():
            if Path(path).name == video_name:
                found_path = path
                break
        
        if found_path:
            video_path = found_path
        else:
            print(f"Video not found in scores!")
            print(f"Looking for: {video_path}")
            print(f"Available videos:")
            for path in scores.keys():
                print(f"  - {path}")
            return
    
    # Get the scores for this video
    frame_scores = scores[video_path]
    video_name = Path(video_path).name
    
    # Extract data
    frames = list(range(len(frame_scores)))
    fish_probs = [fish_prob for _, fish_prob in frame_scores]
    threshold = 0.977989
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot fish probability
    plt.plot(frames, fish_probs, 'b-', linewidth=1, alpha=0.8, label='Fish Probability')
    
    # Add threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Detection Threshold ({threshold:.3f})')
    
    # Highlight detected fish frames
    fish_frames = [(i, prob) for i, (_, prob) in enumerate(frame_scores) if prob >= threshold]
    if fish_frames:
        fish_frame_nums, fish_frame_probs = zip(*fish_frames)
        plt.scatter(fish_frame_nums, fish_frame_probs, color='red', s=20, alpha=0.7, 
                   label=f'Fish Detected ({len(fish_frames)} frames)', zorder=5)
    
    # Formatting
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Fish Probability', fontsize=12)
    plt.title(f'Fish Detection Scores: {video_name}', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistics
    total_frames = len(frame_scores)
    fish_count = len(fish_frames) if fish_frames else 0
    fish_percentage = (fish_count / total_frames) * 100
    max_prob = max(fish_probs)
    avg_prob = np.mean(fish_probs)
    
    stats_text = f'Total Frames: {total_frames:,}\n'
    stats_text += f'Fish Detected: {fish_count:,} ({fish_percentage:.1f}%)\n'
    stats_text += f'Max Probability: {max_prob:.3f}\n'
    stats_text += f'Average Probability: {avg_prob:.3f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPlotted scores for: {video_name}")
    print(f"Total frames: {total_frames:,}")
    print(f"Fish detected: {fish_count:,} frames ({fish_percentage:.1f}%)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_plot.py /path/to/your/video.mp4")
        print("\nExample:")
        print("  python simple_plot.py video1.mp4")
        print("  python simple_plot.py /Users/me/Videos/my_video.mp4")
        return
    
    video_path = sys.argv[1]
    plot_fish_scores(video_path)

if __name__ == "__main__":
    main()