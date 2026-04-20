# Timeline Viewer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the clunky single-file HTML timeline viewer with a three-panel SPA (sidebar + player + unified timeline) that separates data (JSON) from presentation (HTML/CSS/JS).

**Architecture:** Python generates `viewer_data.json` merging single + multi detection results, copies videos, and copies the static `viewer/` app to the output directory. The viewer app (`index.html`, `style.css`, `app.js`) loads the JSON at startup and renders the three-panel UI with keyboard navigation.

**Tech Stack:** Python 3 (existing), vanilla HTML/CSS/JS, existing HTTP server from `fish_timeline_viewer.py`.

---

### File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `viewer/index.html` | Create | App shell — HTML structure for top bar, sidebar, player, timeline, controls |
| `viewer/style.css` | Create | Dark theme styles for all components |
| `viewer/app.js` | Create | Data loading, video switching, timeline rendering, keyboard shortcuts, playhead sync |
| `fish_timeline_viewer.py` | Rewrite | Generate `viewer_data.json`, copy videos, copy viewer app, serve via HTTP |

---

### Task 1: Create the HTML app shell

**Files:**
- Create: `viewer/index.html`

- [ ] **Step 1: Create viewer directory**

```bash
mkdir -p viewer
```

- [ ] **Step 2: Write index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fish Detection Viewer</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <!-- Top Bar -->
    <header id="top-bar">
        <div class="top-bar-left">
            <span class="logo">🐟 Fish Detection Viewer</span>
        </div>
        <div class="top-bar-right">
            <span id="stat-videos">📹 0 videos</span>
            <span id="stat-single" class="stat-single">● 0 single</span>
            <span id="stat-multi" class="stat-multi">● 0 multi</span>
            <span id="stat-session" class="stat-session"></span>
        </div>
    </header>

    <div id="app">
        <!-- Left Sidebar -->
        <aside id="sidebar">
            <div class="sidebar-header">Videos — sorted by detections ↓</div>
            <div id="video-list"></div>
        </aside>

        <!-- Main Area -->
        <main id="main">
            <!-- Video Player -->
            <div id="player-container">
                <video id="video-player" controls preload="metadata"></video>
                <div id="player-overlay" class="player-overlay">
                    <span>Select a video from the sidebar</span>
                </div>
            </div>

            <!-- Unified Timeline -->
            <div id="timeline-section">
                <div class="timeline-header">
                    <span class="timeline-label">TIMELINE</span>
                    <div class="timeline-legend">
                        <span class="legend-single">● single fish</span>
                        <span class="legend-multi">● multi fish</span>
                    </div>
                </div>
                <div id="timeline-bar">
                    <div id="playhead"></div>
                    <div id="timeline-markers"></div>
                </div>
                <div id="timeline-times">
                    <span>0:00</span>
                    <span id="timeline-end">0:00</span>
                </div>
            </div>

            <!-- Controls Bar -->
            <div id="controls-bar">
                <div class="controls-group">
                    <button id="btn-prev-detection" title="Previous detection (P)">⏮ Prev detection</button>
                    <button id="btn-next-detection" title="Next detection (N)">⏭ Next detection</button>
                </div>
                <div class="controls-group">
                    <button id="btn-prev-video" title="Previous video (Shift+P)">⏮ Prev video</button>
                    <button id="btn-next-video" title="Next video (Shift+N)">⏭ Next video</button>
                </div>
                <div class="controls-hints">
                    [P] prev · [N] next · [Space] play/pause · [←→] seek
                </div>
            </div>
        </main>
    </div>

    <!-- Tooltip (positioned absolutely, shown on marker hover) -->
    <div id="marker-tooltip" class="marker-tooltip" style="display:none;">
        <div id="tooltip-time"></div>
        <div id="tooltip-frame"></div>
        <div id="tooltip-prob"></div>
    </div>

    <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Verify file was created**

```bash
cat viewer/index.html | head -5
```

Expected: `<!DOCTYPE html>` and the title line.

- [ ] **Step 4: Commit**

```bash
git add viewer/index.html
git commit -m "feat: add HTML app shell for timeline viewer redesign"
```

---

### Task 2: Create the dark theme stylesheet

**Files:**
- Create: `viewer/style.css`

- [ ] **Step 1: Write style.css**

```css
/* === Reset & Base === */
*, *::before, *::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --bg: #1a1a2e;
    --sidebar-bg: #16213e;
    --controls-bg: #0f3460;
    --accent: #4cc9f0;
    --single: #e74c3c;
    --multi: #f39c12;
    --text: #e0e0e0;
    --text-dim: #888;
    --text-muted: #555;
    --border: #0f3460;
    --timeline-bg: #0a0a1a;
}

html, body {
    height: 100%;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
}

/* === Top Bar === */
#top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    background: var(--sidebar-bg);
    border-bottom: 1px solid var(--border);
    height: 48px;
    flex-shrink: 0;
}

.logo {
    font-weight: bold;
    color: var(--accent);
    font-size: 15px;
}

.top-bar-right {
    display: flex;
    gap: 18px;
    font-size: 12px;
    color: var(--text-dim);
}

.stat-single { color: var(--single); }
.stat-multi { color: var(--multi); }
.stat-session { color: var(--accent); }

/* === Layout === */
#app {
    display: flex;
    height: calc(100vh - 48px);
}

/* === Sidebar === */
#sidebar {
    width: 260px;
    flex-shrink: 0;
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-header {
    padding: 10px 14px;
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}

#video-list {
    overflow-y: auto;
    flex: 1;
}

.video-item {
    padding: 10px 14px;
    cursor: pointer;
    border-bottom: 1px solid rgba(15, 52, 96, 0.5);
    transition: background 0.15s;
}

.video-item:hover {
    background: rgba(76, 201, 240, 0.08);
}

.video-item.active {
    background: var(--controls-bg);
    border-left: 3px solid var(--accent);
    padding-left: 11px;
}

.video-item-name {
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.video-item-stats {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 3px;
}

.video-item-stats .single-count { color: var(--single); }
.video-item-stats .multi-count { color: var(--multi); }

.video-item.no-detections {
    opacity: 0.4;
}

.video-item.no-detections .video-item-name {
    color: var(--text-dim);
}

/* === Main Area === */
#main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* === Player === */
#player-container {
    flex: 1;
    background: #000;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
}

#video-player {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.player-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 18px;
    pointer-events: none;
}

.player-overlay.hidden {
    display: none;
}

/* === Timeline === */
#timeline-section {
    background: var(--sidebar-bg);
    padding: 10px 18px 6px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
}

.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.timeline-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.timeline-legend {
    font-size: 10px;
    display: flex;
    gap: 12px;
}

.legend-single { color: var(--single); }
.legend-multi { color: var(--multi); }

#timeline-bar {
    position: relative;
    height: 32px;
    background: var(--timeline-bg);
    border-radius: 4px;
    cursor: pointer;
    overflow: visible;
}

#playhead {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--accent);
    z-index: 3;
    pointer-events: none;
    transition: left 0.1s linear;
}

#timeline-markers {
    position: absolute;
    inset: 0;
}

.timeline-marker {
    position: absolute;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    z-index: 2;
}

.timeline-marker:hover {
    transform: scale(1.6);
    box-shadow: 0 0 8px rgba(255, 255, 255, 0.3);
    z-index: 5;
}

.timeline-marker.highlight {
    transform: scale(1.6);
    box-shadow: 0 0 10px var(--accent);
}

.timeline-marker.single {
    background: var(--single);
    top: 4px;
}

.timeline-marker.multi {
    background: var(--multi);
    bottom: 4px;
    top: auto;
}

#timeline-times {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: var(--text-muted);
    margin-top: 3px;
}

/* === Marker Tooltip === */
.marker-tooltip {
    position: fixed;
    background: #2c3e50;
    color: #fff;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 11px;
    line-height: 1.4;
    pointer-events: none;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

/* === Controls Bar === */
#controls-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 18px;
    background: var(--controls-bg);
    flex-shrink: 0;
    border-top: 1px solid var(--border);
}

.controls-group {
    display: flex;
    gap: 8px;
}

#controls-bar button {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: var(--text-dim);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
}

#controls-bar button:hover {
    background: rgba(76, 201, 240, 0.15);
    color: var(--text);
}

.controls-hints {
    color: var(--text-muted);
    font-size: 10px;
}
```

- [ ] **Step 2: Commit**

```bash
git add viewer/style.css
git commit -m "feat: add dark theme stylesheet for timeline viewer"
```

---

### Task 3: Create the application JavaScript

**Files:**
- Create: `viewer/app.js`

This is the core logic. It has four sections: data loading, sidebar rendering, video/timeline management, and keyboard shortcuts.

- [ ] **Step 1: Write app.js**

```javascript
/* === Fish Detection Viewer — app.js === */

(function () {
    "use strict";

    // --- State ---
    let viewerData = null;       // Parsed viewer_data.json
    let videoOrder = [];         // Video filenames sorted by detection count desc
    let currentVideoName = null; // Currently loaded video filename
    let currentDetections = [];  // Merged & sorted detections for current video
    let currentDetectionIdx = -1;// Index into currentDetections (-1 = none)

    // --- DOM refs ---
    const videoPlayer = document.getElementById("video-player");
    const playerOverlay = document.getElementById("player-overlay");
    const videoListEl = document.getElementById("video-list");
    const timelineMarkers = document.getElementById("timeline-markers");
    const playhead = document.getElementById("playhead");
    const timelineBar = document.getElementById("timeline-bar");
    const timelineEnd = document.getElementById("timeline-end");
    const tooltip = document.getElementById("marker-tooltip");
    const tooltipTime = document.getElementById("tooltip-time");
    const tooltipFrame = document.getElementById("tooltip-frame");
    const tooltipProb = document.getElementById("tooltip-prob");

    // Stat elements
    const statVideos = document.getElementById("stat-videos");
    const statSingle = document.getElementById("stat-single");
    const statMulti = document.getElementById("stat-multi");
    const statSession = document.getElementById("stat-session");

    // Buttons
    const btnPrevDet = document.getElementById("btn-prev-detection");
    const btnNextDet = document.getElementById("btn-next-detection");
    const btnPrevVid = document.getElementById("btn-prev-video");
    const btnNextVid = document.getElementById("btn-next-video");

    // --- Helpers ---
    function formatTime(seconds) {
        var m = Math.floor(seconds / 60);
        var s = Math.floor(seconds % 60);
        return m + ":" + (s < 10 ? "0" : "") + s;
    }

    // --- Data Loading ---
    function loadData() {
        fetch("viewer_data.json")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                viewerData = data;
                init();
            })
            .catch(function (err) {
                playerOverlay.querySelector("span").textContent =
                    "Failed to load viewer_data.json: " + err.message;
            });
    }

    function init() {
        // Compute totals and sort videos
        var totalSingle = 0;
        var totalMulti = 0;
        var videoEntries = [];

        Object.keys(viewerData.videos).forEach(function (name) {
            var v = viewerData.videos[name];
            var sc = v.single_detections ? v.single_detections.length : 0;
            var mc = v.multi_detections ? v.multi_detections.length : 0;
            totalSingle += sc;
            totalMulti += mc;
            videoEntries.push({ name: name, single: sc, multi: mc, total: sc + mc });
        });

        // Sort descending by total detections
        videoEntries.sort(function (a, b) { return b.total - a.total; });
        videoOrder = videoEntries.map(function (e) { return e.name; });

        // Update top bar stats
        statVideos.textContent = "📹 " + videoOrder.length + " videos";
        statSingle.textContent = "● " + totalSingle + " single";
        statMulti.textContent = "● " + totalMulti + " multi";
        statSession.textContent = "Session: " + (viewerData.session || "");

        // Render sidebar
        renderSidebar(videoEntries);

        // Auto-select first video with detections
        var firstWithDetections = videoEntries.find(function (e) { return e.total > 0; });
        if (firstWithDetections) {
            loadVideo(firstWithDetections.name);
        } else if (videoOrder.length > 0) {
            loadVideo(videoOrder[0]);
        }
    }

    // --- Sidebar ---
    function renderSidebar(videoEntries) {
        videoListEl.innerHTML = "";
        videoEntries.forEach(function (entry) {
            var div = document.createElement("div");
            div.className = "video-item" + (entry.total === 0 ? " no-detections" : "");
            div.dataset.video = entry.name;

            var displayName = entry.name.replace(/\.(mp4|MP4)$/, "");

            var statsHtml = "";
            if (entry.total > 0) {
                statsHtml = '<span class="single-count">● ' + entry.single + ' single</span>';
                if (entry.multi > 0) {
                    statsHtml += ' · <span class="multi-count">● ' + entry.multi + ' multi</span>';
                } else {
                    statsHtml += ' · <span>no multi</span>';
                }
            } else {
                statsHtml = '<span>no detections</span>';
            }

            div.innerHTML =
                '<div class="video-item-name" title="' + entry.name + '">' + displayName + '</div>' +
                '<div class="video-item-stats">' + statsHtml + '</div>';

            div.addEventListener("click", function () {
                loadVideo(entry.name);
            });

            videoListEl.appendChild(div);
        });
    }

    function updateSidebarActive() {
        var items = videoListEl.querySelectorAll(".video-item");
        items.forEach(function (item) {
            if (item.dataset.video === currentVideoName) {
                item.classList.add("active");
                item.scrollIntoView({ block: "nearest" });
            } else {
                item.classList.remove("active");
            }
        });
    }

    // --- Video Loading ---
    function loadVideo(videoName) {
        if (!viewerData || !viewerData.videos[videoName]) return;

        currentVideoName = videoName;
        var vdata = viewerData.videos[videoName];

        // Update video source
        videoPlayer.src = "videos/" + videoName;
        playerOverlay.classList.add("hidden");

        // Merge detections sorted by time
        currentDetections = [];
        (vdata.single_detections || []).forEach(function (d) {
            currentDetections.push({
                time: d.time,
                frame: d.frame,
                probability: d.probability,
                type: "single"
            });
        });
        (vdata.multi_detections || []).forEach(function (d) {
            currentDetections.push({
                time: d.time,
                frame: d.frame,
                probability: d.probability,
                type: "multi"
            });
        });
        currentDetections.sort(function (a, b) { return a.time - b.time; });
        currentDetectionIdx = -1;

        // Update timeline
        renderTimeline(vdata.duration);

        // Update sidebar highlight
        updateSidebarActive();

        // Update timeline end time
        timelineEnd.textContent = formatTime(vdata.duration);

        // Seek to first detection if available
        videoPlayer.addEventListener("loadedmetadata", function handler() {
            videoPlayer.removeEventListener("loadedmetadata", handler);
            if (currentDetections.length > 0) {
                currentDetectionIdx = 0;
                videoPlayer.currentTime = currentDetections[0].time;
                highlightCurrentDetection();
            }
        });
    }

    // --- Timeline ---
    function renderTimeline(duration) {
        timelineMarkers.innerHTML = "";
        if (duration <= 0) return;

        currentDetections.forEach(function (det, idx) {
            var pct = (det.time / duration) * 100;
            var marker = document.createElement("div");
            marker.className = "timeline-marker " + det.type;
            marker.style.left = "calc(" + pct + "% - 5px)";
            marker.dataset.index = idx;

            marker.addEventListener("mouseenter", function (e) {
                tooltipTime.textContent = "⏱ " + formatTime(det.time);
                tooltipFrame.textContent = "Frame " + det.frame;
                tooltipProb.textContent = (det.probability * 100).toFixed(1) + "%";
                tooltip.style.display = "block";
                positionTooltip(e);
            });

            marker.addEventListener("mousemove", positionTooltip);

            marker.addEventListener("mouseleave", function () {
                tooltip.style.display = "none";
            });

            timelineMarkers.appendChild(marker);
        });
    }

    function positionTooltip(e) {
        tooltip.style.left = (e.clientX + 12) + "px";
        tooltip.style.top = (e.clientY - 60) + "px";
    }

    function highlightCurrentDetection() {
        // Remove previous highlight
        var prev = timelineMarkers.querySelector(".highlight");
        if (prev) prev.classList.remove("highlight");

        if (currentDetectionIdx >= 0 && currentDetectionIdx < currentDetections.length) {
            var markers = timelineMarkers.querySelectorAll(".timeline-marker");
            if (markers[currentDetectionIdx]) {
                markers[currentDetectionIdx].classList.add("highlight");
            }
        }
    }

    // Timeline bar click — seek to position
    timelineBar.addEventListener("click", function (e) {
        if (!currentVideoName) return;
        var vdata = viewerData.videos[currentVideoName];
        if (!vdata) return;

        // Check if a marker was clicked
        var markerEl = e.target.closest(".timeline-marker");
        if (markerEl) {
            var idx = parseInt(markerEl.dataset.index, 10);
            if (!isNaN(idx) && idx >= 0 && idx < currentDetections.length) {
                currentDetectionIdx = idx;
                seekToTime(currentDetections[idx].time);
                highlightCurrentDetection();
                return;
            }
        }

        // Otherwise, seek to clicked position on the bar
        var rect = timelineBar.getBoundingClientRect();
        var pct = (e.clientX - rect.left) / rect.width;
        var seekTime = pct * vdata.duration;
        seekToTime(seekTime);
    });

    // --- Playhead Sync ---
    videoPlayer.addEventListener("timeupdate", function () {
        if (!currentVideoName || !viewerData) return;
        var vdata = viewerData.videos[currentVideoName];
        if (!vdata || vdata.duration <= 0) return;

        var pct = (videoPlayer.currentTime / vdata.duration) * 100;
        playhead.style.left = pct + "%";
    });

    // --- Seeking ---
    function seekToTime(time) {
        videoPlayer.currentTime = time;
        videoPlayer.play().catch(function () {});
    }

    // --- Detection Navigation ---
    function nextDetection() {
        if (currentDetections.length === 0) return;
        if (currentDetectionIdx < currentDetections.length - 1) {
            currentDetectionIdx++;
        } else {
            currentDetectionIdx = 0; // wrap around
        }
        seekToTime(currentDetections[currentDetectionIdx].time);
        highlightCurrentDetection();
    }

    function prevDetection() {
        if (currentDetections.length === 0) return;
        if (currentDetectionIdx > 0) {
            currentDetectionIdx--;
        } else {
            currentDetectionIdx = currentDetections.length - 1; // wrap around
        }
        seekToTime(currentDetections[currentDetectionIdx].time);
        highlightCurrentDetection();
    }

    // --- Video Navigation ---
    function nextVideo() {
        if (videoOrder.length === 0) return;
        var idx = videoOrder.indexOf(currentVideoName);
        var next = (idx + 1) % videoOrder.length;
        loadVideo(videoOrder[next]);
    }

    function prevVideo() {
        if (videoOrder.length === 0) return;
        var idx = videoOrder.indexOf(currentVideoName);
        var prev = (idx - 1 + videoOrder.length) % videoOrder.length;
        loadVideo(videoOrder[prev]);
    }

    // --- Button Handlers ---
    btnNextDet.addEventListener("click", nextDetection);
    btnPrevDet.addEventListener("click", prevDetection);
    btnNextVid.addEventListener("click", nextVideo);
    btnPrevVid.addEventListener("click", prevVideo);

    // --- Keyboard Shortcuts ---
    document.addEventListener("keydown", function (e) {
        // Don't capture when typing in an input
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

        switch (e.key) {
            case "n":
                if (e.shiftKey) { nextVideo(); } else { nextDetection(); }
                e.preventDefault();
                break;
            case "p":
                if (e.shiftKey) { prevVideo(); } else { prevDetection(); }
                e.preventDefault();
                break;
            case "N":
                nextVideo();
                e.preventDefault();
                break;
            case "P":
                prevVideo();
                e.preventDefault();
                break;
            case " ":
                if (videoPlayer.paused) {
                    videoPlayer.play().catch(function () {});
                } else {
                    videoPlayer.pause();
                }
                e.preventDefault();
                break;
            case "ArrowLeft":
                videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5);
                e.preventDefault();
                break;
            case "ArrowRight":
                videoPlayer.currentTime = Math.min(
                    videoPlayer.duration || 0,
                    videoPlayer.currentTime + 5
                );
                e.preventDefault();
                break;
        }
    });

    // --- Start ---
    loadData();
})();
```

- [ ] **Step 2: Commit**

```bash
git add viewer/app.js
git commit -m "feat: add application logic for timeline viewer (data loading, timeline, keyboard nav)"
```

---

### Task 4: Rewrite fish_timeline_viewer.py

**Files:**
- Rewrite: `fish_timeline_viewer.py`

Replace the old HTML-generating script with one that generates `viewer_data.json`, copies videos, copies the `viewer/` app, and serves everything.

- [ ] **Step 1: Write the new fish_timeline_viewer.py**

```python
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
                "probability": round(det["probability"], 4),
            })

        # Build multi detections
        multi_detections = []
        multi_data = multi_results.get(video_path_str, {})
        for det in multi_data.get("multi_fish_frames", []):
            frame = det["frame"]
            multi_detections.append({
                "frame": frame,
                "time": round(frame / fps, 2),
                "probability": round(det["probability"], 4),
            })

        videos[video_name] = {
            "duration": round(duration, 2),
            "fps": round(fps, 1),
            "single_detections": single_detections,
            "multi_detections": multi_detections,
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
    Copy videos, viewer app, and viewer_data.json into an output directory.
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

    # Copy videos
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    copied = 0
    for video_name, vdata in viewer_data["videos"].items():
        src_path = Path(vdata["source_path"])
        dst_path = videos_dir / video_name
        if src_path.exists() and not dst_path.exists():
            print(f"  Copying {video_name}...")
            shutil.copy2(src_path, dst_path)
            copied += 1
        elif dst_path.exists():
            print(f"  {video_name} already copied, skipping.")

    print(f"Copied {copied} new video(s) to {videos_dir}")
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
    """Start HTTP server serving from directory."""
    os.chdir(directory)

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", port), QuietHandler)
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
```

- [ ] **Step 2: Commit**

```bash
git add fish_timeline_viewer.py
git commit -m "feat: rewrite timeline viewer to generate JSON + serve static SPA"
```

---

### Task 5: Manual integration test

**Files:** None (testing only)

- [ ] **Step 1: Run the viewer against the existing session**

```bash
python fish_timeline_viewer.py --session 14_08_2025
```

Expected: browser opens at `http://localhost:8000` (or similar port). You should see:
- Top bar with session name and detection counts
- Sidebar with videos sorted by detection count
- First video with detections auto-loaded and paused at first detection
- Red dots on timeline for single-fish, orange for multi
- Clicking a dot jumps the video
- N/P keys navigate between detections
- Shift+N / Shift+P navigate between videos
- Space toggles play/pause
- Arrow keys seek ±5s

- [ ] **Step 2: Verify videos with no detections appear dimmed in sidebar**

Scroll down the sidebar. Videos with 0 detections should be semi-transparent.

- [ ] **Step 3: Verify tooltip on marker hover**

Hover over a timeline marker. A tooltip should appear showing time, frame number, and probability.

- [ ] **Step 4: Commit all files together as final integration**

```bash
git add -A
git commit -m "feat: complete timeline viewer redesign — SPA with sidebar, unified timeline, keyboard nav"
```

---

### Task 6: Clean up old HTML generation code

**Files:**
- Modify: `fish_timeline_viewer.py` (already done in Task 4 — this is verification)

- [ ] **Step 1: Verify no references to old `create_working_html` function remain**

```bash
grep -r "create_working_html" .
```

Expected: no matches.

- [ ] **Step 2: Verify old timeline_viewer.html files still exist in output (not deleted)**

The old generated HTML files in `output/detection_output/14_08_2025/fish_detection/timeline_viewer.html` should still be there — we didn't delete them. The new viewer lives in `output/detection_output/14_08_2025/viewer_app/`.

```bash
ls output/detection_output/14_08_2025/fish_detection/timeline_viewer.html
ls output/detection_output/14_08_2025/viewer_app/index.html
```

Expected: both files exist.

- [ ] **Step 3: Commit if any cleanup was needed**

Only if changes were made in this step.
