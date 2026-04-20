# Fish Detection Timeline Viewer Redesign

## Context

The current timeline viewer (`fish_timeline_viewer.py`) generates a single HTML file with embedded CSS/JS. It works but has usability problems:

- Timeline marker clicks are broken (inline `onclick` conflicts with `addEventListener` that calls `stopPropagation`)
- Accordion expand/collapse pattern doesn't scale to 20+ videos
- No way to see single and multi-fish detections together
- No keyboard navigation between detections

The viewer is used for **field monitoring**: operators reviewing 20+ daily video feeds, needing to efficiently jump between fish detections across videos.

## Approach: Lightweight SPA (Vanilla JS + JSON)

Separate data generation (Python) from presentation (static HTML/CSS/JS app). Python outputs a JSON data file; a standalone web app reads and displays it.

### File Structure

```
fish_timeline_viewer.py          # Modified: generates JSON + copies videos + serves app
viewer/
  index.html                     # App shell
  style.css                      # Styles
  app.js                         # All application logic
```

The Python script generates a `viewer_data.json` file alongside the videos, then serves the `viewer/` directory via the existing HTTP server.

### Data Format (viewer_data.json)

```json
{
  "session": "14_08_2025",
  "generated_at": "2026-04-20T12:00:00",
  "videos": {
    "Camera_A_08h30.mp4": {
      "duration": 920.5,
      "fps": 30.0,
      "single_detections": [
        {"frame": 150, "time": 5.0, "probability": 0.985}
      ],
      "multi_detections": [
        {"frame": 420, "time": 14.0, "probability": 0.972}
      ]
    }
  }
}
```

Python merges single and multi results from their respective `results.json` files into this unified format. The `--mode` flag is removed; the viewer always shows all available detection types.

### UI Layout

Three-panel layout (dark theme):

1. **Top bar** — Session name, total video count, total single/multi detection counts.

2. **Left sidebar (240px)** — Scrollable video list sorted by total detection count (descending). Each entry shows video name, single count (red), multi count (orange). Videos with no detections are dimmed. Active video highlighted with accent border. Click to load video.

3. **Main area** (flex column):
   - **Video player** — HTML5 `<video>` element, persistent (not recreated on video switch). Only `src` changes.
   - **Unified timeline** — Horizontal bar. Single-fish markers (red circles) on top row, multi-fish markers (orange circles) on bottom row. Cyan playhead synced to video `timeupdate`. Click marker to seek. Hover shows tooltip with time, frame, probability.
   - **Controls bar** — Prev/Next detection buttons, Prev/Next video buttons, keyboard shortcut hints.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `N` | Jump to next detection (any type) in current video |
| `P` | Jump to previous detection |
| `Space` | Play/pause |
| `←` / `→` | Seek -5s / +5s |
| `Shift+N` | Next video (in sidebar order) |
| `Shift+P` | Previous video |

### Behavior Details

- **Video switching**: Clicking a sidebar entry updates `video.src` to the new file, loads detections for that video, rebuilds the timeline markers. Sidebar selection updates.
- **Detection jumping (N/P)**: All detections (single + multi) are merged into a sorted-by-time array. N/P walks this array and seeks the video. Current detection is highlighted on the timeline.
- **Playhead sync**: `video.timeupdate` event moves a CSS-positioned playhead div across the timeline bar.
- **Marker click**: Sets `video.currentTime`, waits for `seeked` event, then plays. No inline onclick — all event delegation on the timeline container.
- **Initial state**: First video with detections is auto-selected. Video is paused at the first detection timestamp.

### Python Changes

`fish_timeline_viewer.py` modifications:

1. New function `generate_viewer_data()` — reads both `fish_detection/results.json` and `multi_fish/results.json`, merges into the unified JSON format, writes `viewer_data.json`.
2. `create_timeline_viewer()` simplified — no more HTML generation. Copies videos, calls `generate_viewer_data()`, copies `viewer/` directory to output.
3. `--mode` argument removed. Always processes both single and multi if available.
4. HTTP server serves from the output directory (which now contains `index.html`, `style.css`, `app.js`, `viewer_data.json`, and `videos/`).

### Dark Theme Colors

- Background: `#1a1a2e`
- Sidebar: `#16213e`
- Accent: `#4cc9f0` (cyan)
- Single fish markers: `#e74c3c` (red)
- Multi fish markers: `#f39c12` (orange)
- Text primary: `#e0e0e0`
- Text secondary: `#888`

### What's NOT in scope

- No React/Vue/framework — vanilla JS only
- No database or persistence of user state
- No annotation or threshold tuning UI
- No filtering/sorting controls beyond the default sort-by-detections
- No mobile responsiveness (desktop tool)
