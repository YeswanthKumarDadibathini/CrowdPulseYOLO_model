"""
CrowdPulse - Video Stream Processor (Phase 2)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 2 Vision & Detection Pipeline

Beginner-Friendly Explanation:
This module processes video files (like CCTV footage or recorded MP4s).
A video is simply a sequence of images (called frames) played rapidly.

For every frame (or every Nth frame for performance):
1. OpenCV reads the frame
2. Our CrowdDetector runs YOLO person detection
3. We calculate the moving average of people count
4. We track processing FPS (Frames Per Second)
5. We yield the annotated frame so Streamlit can display it live.
"""

import time
from collections import deque
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

import config
from detection.detector import CrowdDetector


class VideoStreamProcessor:
    """Processes video feeds frame-by-frame with YOLO person detection."""

    def __init__(self, detector: Optional[CrowdDetector] = None):
        """Initializes the processor with a shared CrowdDetector instance."""
        self.detector = detector if detector is not None else CrowdDetector()

    def process_video_file(
        self,
        video_path: Union[str, Path],
        confidence_threshold: float = 0.25,
        capacity_limit: Optional[int] = None,
        frame_skip: int = 1,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """Reads a video file and yields (annotated_frame_rgb, frame_stats) for each frame.

        Parameters:
            video_path: Path to MP4, AVI, or MKV file
            confidence_threshold: YOLO confidence cutoff (e.g. 0.25)
            capacity_limit: Maximum safe occupancy for the monitored zone
            frame_skip: Process every Nth frame (1 = process every frame)
            max_frames: Stop after this many frames (None = until video ends)

        Yields:
            tuple: (annotated_frame_rgb, stats_dict)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file at: {video_path}")

        frame_count = 0
        processed_count = 0
        fps_tracker = deque(maxlen=20)
        counts_history = deque(maxlen=30)
        peak_count = 0

        prev_time = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                processed_count += 1
                if max_frames and processed_count > max_frames:
                    break

                # Calculate real-time FPS
                current_time = time.perf_counter()
                fps_calc = 1.0 / max(current_time - prev_time, 1e-5)
                prev_time = current_time
                fps_tracker.append(fps_calc)
                current_fps = sum(fps_tracker) / len(fps_tracker)

                # Convert BGR (OpenCV format) to RGB (Streamlit/Pillow format)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Run YOLO detection
                detection_result = self.detector.detect_people(
                    image_rgb=frame_rgb,
                    confidence_threshold=confidence_threshold,
                    capacity_limit=capacity_limit,
                    draw_annotations=True,
                    fps=current_fps
                )

                person_count = detection_result["count"]
                counts_history.append(person_count)
                peak_count = max(peak_count, person_count)
                avg_count = sum(counts_history) / len(counts_history)

                stats = {
                    "frame_index": frame_count,
                    "processed_index": processed_count,
                    "current_count": person_count,
                    "average_count": round(avg_count, 1),
                    "peak_count": peak_count,
                    "current_fps": round(current_fps, 1),
                    "density_level": detection_result["density_level"],
                    "risk_badge": detection_result["risk_badge"],
                    "occupancy_pct": detection_result["occupancy_pct"],
                    "inference_time_ms": detection_result["inference_time_ms"],
                }

                yield detection_result["annotated_image"], stats

        finally:
            cap.release()


def generate_synthetic_demo_video(
    output_path: Optional[Path] = None,
    num_frames: int = 90,
    fps: int = 15
) -> Path:
    """Creates a lightweight simulated CCTV video of simulated walking crowd silhouettes.

    Useful when no external CCTV MP4 file is available yet.
    """
    if output_path is None:
        output_path = config.VIDEOS_DIR / "demo_concourse.mp4"

    if output_path.exists():
        return output_path

    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize 12 random pedestrian circles
    np.random.seed(42)
    pedestrians = []
    for _ in range(12):
        pedestrians.append({
            "x": float(np.random.randint(50, width - 50)),
            "y": float(np.random.randint(80, height - 80)),
            "vx": float(np.random.uniform(-2.5, 2.5)),
            "vy": float(np.random.uniform(-1.0, 1.0)),
            "radius": int(np.random.randint(12, 18)),
            "color": (int(np.random.randint(180, 240)), int(np.random.randint(180, 240)), 220),
        })

    for f in range(num_frames):
        # Draw concrete concourse floor
        canvas = np.full((height, width, 3), 45, dtype=np.uint8)

        # Draw simulated concourse floor grid lines
        for y in range(0, height, 40):
            cv2.line(canvas, (0, y), (width, y), (55, 55, 55), 1)
        for x in range(0, width, 60):
            cv2.line(canvas, (x, 0), (x, height), (55, 55, 55), 1)

        # Overlay CCTV Timestamp watermark
        timestamp_str = f"CCTV-CAM02 CENTRAL PLAZA [SIMULATION] | FRAME {f:04d}"
        cv2.putText(canvas, timestamp_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Update and draw pedestrians (represented as realistic pedestrian avatars)
        for p in pedestrians:
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Bounce off walls
            if p["x"] <= 30 or p["x"] >= width - 30:
                p["vx"] *= -1
            if p["y"] <= 50 or p["y"] >= height - 40:
                p["vy"] *= -1

            cx, cy = int(p["x"]), int(p["y"])
            r = p["radius"]

            # Draw body (torso and head)
            cv2.circle(canvas, (cx, cy - r), int(r * 0.6), (220, 200, 180), -1)  # Head
            cv2.ellipse(canvas, (cx, cy + int(r * 0.5)), (r, int(r * 1.4)), 0, 0, 360, p["color"], -1) # Body

        writer.write(canvas)

    writer.release()
    return output_path

