


# Fish Detection System


## Authors

- **João da Silva Martins**
Email: joao.da.silva.martins@ahafogvatn.is

An AI-powered system for detecting fish in video files with an interactive timeline viewer. Perfect for researchers, aquaculture professionals, and anyone who needs to analyze fish presence in video footage.

## Features

- **Dual Detection Modes**: Single fish and multiple fish detection
- **AI-Powered Analysis**: Uses advanced vision-language models for accurate detection
- **Interactive Timeline Viewer**: Web-based interface to view videos with fish detection markers
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **GPU Acceleration**: Supports CUDA (NVIDIA) and MPS (Apple Silicon) for faster processing
- **Batch Processing**: Process multiple frames at once
- **Export Results**: JSON format for further analysis

## Quick Start

### 1. Installation

```bash
# Clone or download this repository
git clone <your-repository-url>
cd fish-detection-system
```

### 2. Basic Usage

**Step 1: Detect Fish in Videos**
```bash
# Single fish detection
python fish_detection/fish-detection.py single --video-dir ./your_videos

# Multiple fish detection (requires single detection first)
python fish_detection/fish-detection.py multi --video-dir ./your_videos
```

**Step 2: View Results with Interactive Timeline**
```bash
# Create timeline viewer
python timeline_viewer.py --session your_videos --mode both
```

That's it! Your browser will open with an interactive viewer where you can click on timeline markers to jump directly to fish detections.

## Detailed Usage

### Fish Detection

The system supports two detection modes:

**Single Fish Detection**
```bash
# Process all videos in a directory
python fish_detector.py single --video-dir ./videos

# Process specific videos
python fish_detector.py single --videos video1.mp4 video2.mp4

# Custom output directory
python fish_detector.py single --video-dir ./data --output-dir ./results
```

**Multi-Fish Detection**
```bash
# Must run single detection first!
python fish_detector.py multi --video-dir ./videos
```

### Timeline Viewer

Create interactive web viewers for your detection results:

```bash
# Single fish timeline
python timeline_viewer.py --session your_videos --mode single

# Multi-fish timeline
python timeline_viewer.py --session your_videos --mode multi

# Both modes in separate tabs
python timeline_viewer.py --session your_videos --mode both
```

## How It Works

### Detection Process

1. **Frame Sampling**: Processes every 3rd frame for efficiency
2. **AI Analysis**: Uses OpenCLIP vision-language model with fish-specific prompts
3. **Threshold Filtering**: Only frames above confidence threshold are marked as detections
4. **Results Export**: Saves detailed JSON results and raw probability scores

### Timeline Viewer

1. **Video Copying**: Creates local copies of videos with detections
2. **Web Interface**: Generates HTML viewer with embedded timeline
3. **Interactive Features**: Click timeline markers to jump to exact detection moments
4. **Local Server**: Runs HTTP server for smooth video playback

## Output Structure

```
output/detection_output/
└── your_session_name/
    ├── fish_detection/              # Single fish results
    │   ├── results.json
    │   ├── scores.pkl
    │   ├── videos/                  # Copied videos
    │   └── timeline_viewer.html     # Web viewer
    └── multi_fish/                  # Multi-fish results
        ├── results.json
        ├── scores.pkl
        ├── videos/
        └── timeline_viewer.html
```

## Results Format

### JSON Results Structure
```json
{
  "video_path": {
    "total_frames": 1000,
    "frames_processed": 334,
    "fish_frames": [
      {
        "frame": 45,
        "probability": 0.985
      }
    ]
  }
}
```

## System Requirements

### Hardware
- **CPU**: Any modern processor
- **GPU**: Optional but recommended (NVIDIA CUDA or Apple Silicon)
- **RAM**: 8GB+ recommended
- **Storage**: Space for video copies in output directory

### Software
- **Python**: 3.8 or higher


## Tips for Best Results

### Processing Tips
- Run single fish detection before multi-fish detection
- Use GPU acceleration when available (automatically detected)
- Check the timeline viewer to verify detection quality

## Troubleshooting

### Common Issues

**"No videos found"**
- Check video file format (MP4 supported)
- Verify directory path is correct
- Ensure videos aren't corrupted

**"Model loading failed"**
- Check internet connection (models download on first use)
- Verify sufficient disk space
- Try CPU mode if GPU issues occur

**"Timeline viewer not opening"**
- Check if port 8000 is available
- Try a different browser
- Look for firewall blocking local server

### Performance Optimization

**For Faster Processing:**
- Use NVIDIA GPU with CUDA or Apple Silicon with MPS
- Process smaller batches if memory limited
- Use SSD storage for video files

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure video files are in supported format (MP4)
4. Check system requirements are met

