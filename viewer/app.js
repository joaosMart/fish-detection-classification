/* === Fish Detection Viewer — app.js === */

(function () {
    "use strict";

    // --- State ---
    let viewerData = null;
    let videoOrder = [];
    let currentVideoName = null;
    let currentDetections = [];
    let currentDetectionIdx = -1;

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
    const tooltipSegment = document.getElementById("tooltip-segment");
    const timelineSegments = document.getElementById("timeline-segments");

    const statVideos = document.getElementById("stat-videos");
    const statSingle = document.getElementById("stat-single");
    const statMulti = document.getElementById("stat-multi");
    const statSession = document.getElementById("stat-session");

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
        var totalSingle = 0;
        var totalMulti = 0;
        var videoEntries = [];

        Object.keys(viewerData.videos).forEach(function (name) {
            var v = viewerData.videos[name];
            var sc = v.single_detections ? v.single_detections.length : 0;
            var mc = v.multi_detections ? v.multi_detections.length : 0;
            var segc = v.segments ? v.segments.length : 0;
            totalSingle += sc;
            totalMulti += mc;
            videoEntries.push({ name: name, single: sc, multi: mc, segments: segc, total: sc + mc });
        });

        videoEntries.sort(function (a, b) { return b.total - a.total; });
        videoOrder = videoEntries.map(function (e) { return e.name; });

        statVideos.textContent = "📹 " + videoOrder.length + " videos";
        statSingle.textContent = "● " + totalSingle + " single";
        statMulti.textContent = "● " + totalMulti + " multi";
        statSession.textContent = "Session: " + (viewerData.session || "");

        renderSidebar(videoEntries);

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
                if (entry.segments > 0) {
                    statsHtml += ' · <span class="segment-count">■ ' + entry.segments + ' seg</span>';
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

        videoPlayer.src = "/serve-video?path=" + encodeURIComponent(vdata.source_path);
        playerOverlay.classList.add("hidden");

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

        renderTimeline(vdata.duration);
        updateSidebarActive();
        timelineEnd.textContent = formatTime(vdata.duration);

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
        timelineSegments.innerHTML = "";
        if (duration <= 0) return;

        // Render segment bands
        var vdata = viewerData.videos[currentVideoName];
        var segments = vdata.segments || [];
        segments.forEach(function (seg) {
            var startPct = (seg.start_time / duration) * 100;
            var endPct = (seg.end_time / duration) * 100;
            var widthPct = endPct - startPct;

            var band = document.createElement("div");
            band.className = "segment-band";
            band.style.left = startPct + "%";
            band.style.width = Math.max(widthPct, 0.5) + "%";

            var label = document.createElement("span");
            label.className = "segment-label";
            label.textContent = "S" + seg.segment_number;
            band.appendChild(label);

            timelineSegments.appendChild(band);
        });

        // Build frame-to-segment lookup
        var frameToSegment = {};
        segments.forEach(function (seg) {
            for (var f = seg.start_frame; f <= seg.end_frame; f++) {
                frameToSegment[f] = seg.segment_number;
            }
        });

        currentDetections.forEach(function (det, idx) {
            var pct = (det.time / duration) * 100;
            var marker = document.createElement("div");
            marker.className = "timeline-marker " + det.type;
            marker.style.left = "calc(" + pct + "% - 5px)";
            marker.dataset.index = idx;

            var segNum = frameToSegment[det.frame];

            marker.addEventListener("mouseenter", function (e) {
                tooltipTime.textContent = "⏱ " + formatTime(det.time);
                tooltipFrame.textContent = "Frame " + det.frame;
                tooltipProb.textContent = (det.probability * 100).toFixed(1) + "%";
                tooltipSegment.textContent = segNum ? "Segment " + segNum : "";
                tooltipSegment.style.display = segNum ? "block" : "none";
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
        var prev = timelineMarkers.querySelector(".highlight");
        if (prev) prev.classList.remove("highlight");

        if (currentDetectionIdx >= 0 && currentDetectionIdx < currentDetections.length) {
            var markers = timelineMarkers.querySelectorAll(".timeline-marker");
            if (markers[currentDetectionIdx]) {
                markers[currentDetectionIdx].classList.add("highlight");
            }
        }
    }

    // Timeline bar click
    timelineBar.addEventListener("click", function (e) {
        if (!currentVideoName) return;
        var vdata = viewerData.videos[currentVideoName];
        if (!vdata) return;

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
            currentDetectionIdx = 0;
        }
        seekToTime(currentDetections[currentDetectionIdx].time);
        highlightCurrentDetection();
    }

    function prevDetection() {
        if (currentDetections.length === 0) return;
        if (currentDetectionIdx > 0) {
            currentDetectionIdx--;
        } else {
            currentDetectionIdx = currentDetections.length - 1;
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
