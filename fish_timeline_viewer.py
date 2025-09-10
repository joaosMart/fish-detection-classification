#!/usr/bin/env python3
"""
Fish Detection Timeline Viewer - WORKING VERSION
Copies full videos and creates an interactive web viewer with timeline markers for fish detections.
"""

import os
import sys
import json
import shutil
import argparse
import webbrowser
import threading
import time
from pathlib import Path
from typing import Dict, List, Any
import cv2
from http.server import HTTPServer, SimpleHTTPRequestHandler

def convert_frame_to_seconds(frame_number: int, fps: float) -> float:
    """Convert frame number to seconds"""
    return frame_number / fps

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video information (FPS, duration, etc.)"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            'fps': fps if fps > 0 else 30.0,
            'frame_count': frame_count,
            'duration': duration
        }
    except:
        return {'fps': 30.0, 'frame_count': 0, 'duration': 0}

def load_detection_results(results_file: Path) -> Dict[str, Any]:
    """Load detection results from JSON file"""
    if not results_file.exists():
        return {}
    
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return {}

def create_working_html(viewer_file: Path, 
                        timeline_data: Dict, 
                        title: str,
                        header_color: str,
                        marker_color: str):
    """Generate working HTML viewer based on the simple version that worked"""
    
    total_videos = len(timeline_data)
    total_detections = sum(data['total_detections'] for data in timeline_data.values())
    
    # HTML start with CSS
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>''' + title + '''</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .summary {
            background: linear-gradient(135deg, ''' + header_color + ''', ''' + marker_color + '''dd);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.2em;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .video-container {
            margin: 20px 0;
            background: #f8f9fa;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        .video-header {
            padding: 20px 25px;
            background: linear-gradient(135deg, ''' + header_color + ''', ''' + header_color + '''dd);
            color: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }
        .video-header:hover {
            background: linear-gradient(135deg, ''' + header_color + '''dd, ''' + header_color + '''bb);
        }
        .video-name {
            font-size: 1.3em;
            font-weight: bold;
        }
        .video-stats {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .dropdown-arrow {
            font-size: 1.2em;
            transition: transform 0.3s ease;
            margin-left: 15px;
        }
        .video-content {
            overflow: hidden;
            transition: max-height 0.4s ease, padding 0.4s ease;
            background: white;
            max-height: 0;
            padding: 0 25px;
        }
        .video-content.expanded {
            max-height: 2000px;
            padding: 25px;
        }
        video {
            width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .timeline-container {
            margin: 20px 0;
            position: relative;
        }
        .timeline {
            width: 100%;
            height: 50px;
            background: linear-gradient(to right, #ecf0f1, #bdc3c7);
            border-radius: 25px;
            position: relative;
            margin: 15px 0;
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.1);
        }
        .timeline-marker {
            position: absolute;
            width: 16px;
            height: 16px;
            background: ''' + marker_color + ''';
            border: 2px solid white;
            border-radius: 50%;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .timeline-marker:hover {
            transform: translateY(-50%) scale(1.4);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            z-index: 10;
        }
        .marker-tooltip {
            position: absolute;
            bottom: 25px;
            left: 50%;
            transform: translateX(-50%);
            background: #2c3e50;
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 11px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .timeline-marker:hover .marker-tooltip {
            opacity: 1;
        }
        .marker-tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -4px;
            border: 4px solid transparent;
            border-top-color: #2c3e50;
        }
        .video-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            background: #e8f4fd;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 0.85em;
        }
        .info-item {
            text-align: center;
            padding: 5px;
        }
        .instructions {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 ''' + title + '''</h1>
        
        <div class="summary">
            <div>🎬 ''' + str(total_videos) + ''' Videos | 🐟 ''' + str(total_detections) + ''' Fish Detections</div>
        </div>
        
        <div class="instructions">
            <h3>🎯 How to Use:</h3>
            <p><strong>Click on video names</strong> to expand and view the video with fish detection timeline.</p>
            <p><strong>Click the red dots</strong> in the timeline to jump directly to fish detections!</p>
        </div>
'''

    # Add video sections
    video_count = 0
    for video_name, video_data in timeline_data.items():
        markers = video_data['markers']
        duration = video_data['duration']
        fps = video_data['fps']
        
        # Format duration
        duration_mins = int(duration // 60)
        duration_secs = int(duration % 60)
        duration_str = f"{duration_mins}:{duration_secs:02d}"
        
        # Clean video name and create safe ID
        display_name = video_name.replace('.mp4', '').replace('.MP4', '')
        safe_id = f"video{video_count}"
        
        html_content += '''
        <div class="video-container">
            <div class="video-header" onclick="toggleVideo(''' + "'" + safe_id + "'" + ''')">
                <div class="video-name">🎬 ''' + display_name + '''</div>
                <div>
                    <span class="video-stats">🐟 ''' + str(len(markers)) + ''' detections • ⏱️ ''' + duration_str + '''</span>
                    <span class="dropdown-arrow" id="arrow_''' + safe_id + '''">▼</span>
                </div>
            </div>
            
            <div class="video-content" id="content_''' + safe_id + '''">
                <video id="player_''' + safe_id + '''" controls preload="metadata">
                    <source src="videos/''' + video_name + '''" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                
                <div class="video-info">
                    <div class="info-item"><strong>Duration:</strong><br>''' + duration_str + '''</div>
                    <div class="info-item"><strong>Detections:</strong><br>''' + str(len(markers)) + '''</div>
                    <div class="info-item"><strong>FPS:</strong><br>''' + f"{fps:.1f}" + '''</div>
                </div>
                
                <div class="timeline-container">
                    <div class="timeline">'''
        
        # Add timeline markers
        for i, marker in enumerate(markers):
            percentage = marker['percentage']
            time_seconds = marker['time']
            frame_num = marker['frame']
            probability = marker['probability']
            
            # Format time as MM:SS
            time_mins = int(time_seconds // 60)
            time_secs = int(time_seconds % 60)
            time_str = f"{time_mins}:{time_secs:02d}"
            
            html_content += '''
                        <div class="timeline-marker" 
                             style="left: ''' + f"{percentage:.1f}" + '''%"
                             onclick="jumpToTime(''' + "'" + f"player_{safe_id}" + "'" + ''', ''' + str(time_seconds) + ''')">
                            <div class="marker-tooltip">
                                ''' + time_str + '''<br>
                                Frame ''' + str(frame_num) + '''<br>
                                ''' + f"{probability:.1%}" + '''
                            </div>
                        </div>'''
        
        html_content += '''
                    </div>
                </div>
            </div>
        </div>'''
        
        video_count += 1

    # Add JavaScript and closing HTML
    html_content += '''
    </div>
    
    <script>
        function toggleVideo(videoId) {
            console.log('Toggling video:', videoId);
            
            var content = document.getElementById('content_' + videoId);
            var arrow = document.getElementById('arrow_' + videoId);
            
            console.log('Content element:', content);
            console.log('Arrow element:', arrow);
            
            if (content && arrow) {
                if (content.classList.contains('expanded')) {
                    // Collapse
                    content.classList.remove('expanded');
                    arrow.style.transform = 'rotate(0deg)';
                    arrow.textContent = '▼';
                    console.log('Video collapsed');
                } else {
                    // Expand
                    content.classList.add('expanded');
                    arrow.style.transform = 'rotate(180deg)';
                    arrow.textContent = '▲';
                    console.log('Video expanded');
                    
                    // Scroll into view after expansion
                    setTimeout(function() {
                        content.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 200);
                }
            } else {
                console.error('Could not find elements for videoId:', videoId);
            }
        }
        
        function jumpToTime(videoId, timeInSeconds) {
            console.log('Jump to time called:', videoId, timeInSeconds);
            var video = document.getElementById(videoId);
            if (video) {
                video.pause();
                
                if (video.readyState < 2) {
                    video.addEventListener('loadeddata', function handler() {
                        video.removeEventListener('loadeddata', handler);
                        seekAndPlay(video, timeInSeconds);
                    });
                } else {
                    seekAndPlay(video, timeInSeconds);
                }
            } else {
                console.error('Video not found:', videoId);
            }
        }
        
        function seekAndPlay(video, timeInSeconds) {
            video.currentTime = timeInSeconds;
            
            video.addEventListener('seeked', function handler() {
                video.removeEventListener('seeked', handler);
                video.play().catch(function(e) {
                    console.log('Play prevented:', e);
                });
            });
            
            setTimeout(function() {
                if (Math.abs(video.currentTime - timeInSeconds) > 0.5) {
                    video.currentTime = timeInSeconds;
                    video.play().catch(function(e) {
                        console.log('Play prevented:', e);
                    });
                }
            }, 1000);
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded');
            console.log('Video containers found:', document.querySelectorAll('.video-container').length);
            console.log('Video headers found:', document.querySelectorAll('.video-header').length);
            console.log('Video contents found:', document.querySelectorAll('.video-content').length);
            console.log('Dropdown arrows found:', document.querySelectorAll('.dropdown-arrow').length);
            
            // Make sure all videos start collapsed
            document.querySelectorAll('.video-content').forEach(function(content) {
                content.classList.remove('expanded');
            });
            
            var markers = document.querySelectorAll('.timeline-marker');
            markers.forEach(function(marker) {
                marker.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    this.style.background = '#27ae60';
                    this.style.transform = 'translateY(-50%) scale(1.3)';
                    
                    var self = this;
                    setTimeout(function() {
                        self.style.background = ''' + "'" + marker_color + "'" + ''';
                        self.style.transform = 'translateY(-50%) scale(1)';
                    }, 600);
                });
            });
        });
        
        document.addEventListener('keydown', function(e) {
            var videos = document.querySelectorAll('video');
            videos.forEach(function(video) {
                if (e.key === ' ' && document.activeElement === video) {
                    e.preventDefault();
                    if (video.paused) {
                        video.play();
                    } else {
                        video.pause();
                    }
                }
            });
        });
    </script>
</body>
</html>'''

    # Write the complete HTML
    with open(viewer_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

def create_timeline_viewer(session_dir: Path, mode: str):
    """Create timeline viewer with fish detection markers"""
    
    # Determine paths
    if mode == "single":
        results_file = session_dir / "fish_detection" / "results.json"
        videos_dir = session_dir / "fish_detection" / "videos"
        viewer_file = session_dir / "fish_detection" / "timeline_viewer.html"
        fish_key = "fish_frames"
        title = "Single Fish Detection Timeline"
        head_color = "#0599ee"
        mark_color = "#a02424"
    elif mode == "multi":
        results_file = session_dir / "multi_fish" / "results.json"
        videos_dir = session_dir / "multi_fish" / "videos"
        viewer_file = session_dir / "multi_fish" / "timeline_viewer.html"
        fish_key = "multi_fish_frames"
        title = "Multi Fish Detection Timeline"
        head_color = "#0599ee"
        mark_color = "#a02424"
    else:
        print(f"Invalid mode: {mode}")
        return None, None
    
    # Load results
    results = load_detection_results(results_file)
    if not results:
        print(f"No detection results found for {mode} mode")
        return None, None
    
    # Create videos directory
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy videos and prepare timeline data
    timeline_data = {}
    videos_copied = 0
    
    print(f"\n{title} - Processing videos...")
    
    for video_path, detection_data in results.items():
        fish_detections = detection_data.get(fish_key, [])
        if not fish_detections:
            continue
        
        # Check if original video exists
        original_video = Path(video_path)
        if not original_video.exists():
            print(f"Warning: Original video not found: {video_path}")
            continue
        
        # Copy video
        video_name = original_video.name
        output_video_path = videos_dir / video_name
        
        print(f"📋 Processing {video_name}... ({len(fish_detections)} fish detections)")
        shutil.copy2(original_video, output_video_path)
        videos_copied += 1
        
        # Get video info
        video_info = get_video_info(str(original_video))
        fps = video_info['fps']
        duration = video_info['duration']
        
        # Create timeline markers
        markers = []
        for detection in fish_detections:
            frame_num = detection['frame']
            probability = detection['probability']
            time_seconds = convert_frame_to_seconds(frame_num, fps)
            
            markers.append({
                'time': time_seconds,
                'frame': frame_num,
                'probability': probability,
                'percentage': (time_seconds / duration * 100) if duration > 0 else 0
            })
        
        timeline_data[video_name] = {
            'markers': markers,
            'duration': duration,
            'fps': fps,
            'total_detections': len(fish_detections)
        }
    
    if videos_copied == 0:
        print("No videos with fish found!")
        return None, None
    
    # Generate HTML viewer
    create_working_html(viewer_file, timeline_data, title, head_color, mark_color)
    
    print(f"✅ Processed {videos_copied} videos")
    print(f"📁 Videos copied to: {videos_dir}")
    print(f"🌐 Timeline viewer: {viewer_file}")
    
    return viewer_file, videos_dir

def start_local_server(directory: Path, port: int = 8000):
    """Start a local HTTP server to serve video files"""
    os.chdir(directory.parent)
    
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress logs
    
    server = HTTPServer(('localhost', port), QuietHandler)
    print(f"🌐 Server running at http://localhost:{port}")
    server.serve_forever()

def find_available_port(start_port: int = 8000) -> int:
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return start_port

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='🐟 Fish Detection Timeline Viewer - Interactive video timeline with fish markers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --session experiment_videos
  %(prog)s --session experiment_videos --mode multi
  %(prog)s --session experiment_videos --mode both
        """
    )
    
    parser.add_argument('--output-dir', default='./output/detection_output',
                       help='Base output directory (default: ./output/detection_output)')
    
    parser.add_argument('--session', required=True,
                       help='Session name (directory name containing detection results)')
    
    parser.add_argument('--mode', choices=['single', 'multi', 'both'], default='single',
                       help='Detection mode to process (default: single)')
    
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t automatically open browser')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    args = parse_arguments()
    
    print("🐟 Fish Detection Timeline Viewer 🐟")
    print(f"Session: {args.session}")
    print(f"Mode: {args.mode}")
    print("-" * 50)
    
    # Determine session directory
    output_base_dir = Path(args.output_dir)
    session_dir = output_base_dir / args.session
    
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}")
        available_sessions = [item.name for item in output_base_dir.glob("*") if item.is_dir()]
        if available_sessions:
            print("Available sessions:", ", ".join(available_sessions))
        sys.exit(1)
    
    # Create timeline viewer(s)
    viewer_files = []
    
    if args.mode == 'both':
        # Process both single and multi
        for mode in ['single', 'multi']:
            viewer_file, videos_dir = create_timeline_viewer(session_dir, mode)
            if viewer_file:
                viewer_files.append((viewer_file, videos_dir))
    else:
        viewer_file, videos_dir = create_timeline_viewer(session_dir, args.mode)
        if viewer_file:
            viewer_files.append((viewer_file, videos_dir))
    
    if not viewer_files:
        print("❌ Failed to create any viewers!")
        sys.exit(1)
    
    # Start server and open browser for the first viewer
    if not args.no_browser and viewer_files:
        viewer_file, videos_dir = viewer_files[0]
        port = find_available_port()
        server_thread = threading.Thread(target=start_local_server, args=(videos_dir, port), daemon=True)
        server_thread.start()
        
        time.sleep(1)  # Give server time to start
        
        # Open browser
        viewer_url = f"http://localhost:{port}/{viewer_file.name}"
        print(f"🚀 Opening browser: {viewer_url}")
        webbrowser.open(viewer_url)
        
        # Open additional viewers in new tabs
        for viewer_file, _ in viewer_files[1:]:
            time.sleep(0.5)
            viewer_url = f"http://localhost:{port}/{viewer_file.name}"
            webbrowser.open(viewer_url)
        
        print("\n" + "="*50)
        print("🎉 TIMELINE VIEWER IS READY!")
        print("="*50)
        print("• Click video names to expand/collapse")
        print("• Click the red dots on the timeline to jump to fish!")
        print("• Each dot shows exact time, frame, and confidence")
        print("• Videos play from the moment you click")
        print("• Press Ctrl+C to stop the server when done")
        print("="*50)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped!")
    else:
        for viewer_file, _ in viewer_files:
            print(f"📁 Timeline viewer ready: {viewer_file}")

if __name__ == "__main__":
    main()