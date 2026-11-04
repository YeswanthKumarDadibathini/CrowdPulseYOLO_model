"""
CrowdPulse - Crowd Detection & People Counter (Phase 2)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 2 Vision & Detection Pipeline

Beginner-Friendly Explanation:
This module uses YOLOv8 (You Only Look Once) to locate people in images or video frames.
YOLO is a deep learning computer vision model. It scans the entire image in one pass
and predicts bounding boxes [x1, y1, x2, y2] for objects it recognizes.

In the COCO dataset, class 0 represents 'person'.
We filter detections to only class 0, count the detected people, compare against safe
capacity limits from config.py, and draw visual warnings (Green, Yellow, Red).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config


class CrowdDetector:
    """YOLO-powered Crowd Density and Person Detection Engine."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initializes the detector and loads YOLO model weights."""
        if model_path is None:
            # Check models/ directory first, then root folder
            primary = config.MODELS_DIR / "yolov8n.pt"
            fallback = config.BASE_DIR / "yolov8n.pt"
            self.model_path = primary if primary.exists() else fallback
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found at {self.model_path}. "
                f"Ensure 'yolov8n.pt' is present in the models/ directory."
            )

        # Device selection: automatically uses GPU if CUDA is available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        # Warm up the model once with a blank frame to prime caches
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, classes=[0], verbose=False)

    def classify_density(
        self,
        count: int,
        capacity_limit: Optional[int] = None
    ) -> Tuple[str, str, Tuple[int, int, int]]:
        """Determines the risk classification based on people count and capacity limit.

        Returns:
            tuple: (density_level_str, risk_badge_str, rgb_color_tuple)
        """
        if capacity_limit and capacity_limit > 0:
            occupancy_pct = (count / capacity_limit) * 100.0
            if occupancy_pct < config.DENSITY_THRESHOLDS["NORMAL"]:
                return "NORMAL", "🟢 Safe", (46, 204, 113)      # Emerald Green
            elif occupancy_pct < config.DENSITY_THRESHOLDS["MODERATE"]:
                return "MODERATE", "🟡 Caution", (241, 196, 15)   # Amber Yellow
            else:
                return "CRITICAL", "🔴 Overcrowded", (231, 76, 60) # Crimson Red
        else:
            # Generic threshold when no specific zone capacity is given
            if count <= 15:
                return "NORMAL", "🟢 Low Density", (46, 204, 113)
            elif count <= 40:
                return "MODERATE", "🟡 Medium Density", (241, 196, 15)
            else:
                return "CRITICAL", "🔴 High Density", (231, 76, 60)

    def detect_people(
        self,
        image_rgb: np.ndarray,
        confidence_threshold: float = 0.25,
        capacity_limit: Optional[int] = None,
        draw_annotations: bool = True,
        fps: Optional[float] = None
    ) -> Dict:
        """Runs YOLO person detection on an RGB image.

        Parameters:
            image_rgb (np.ndarray): Input image in RGB format (H, W, 3)
            confidence_threshold (float): Minimum confidence score (0.0 to 1.0)
            capacity_limit (int, optional): Safe capacity of the monitored area
            draw_annotations (bool): Whether to draw boxes and HUD on the image
            fps (float, optional): Current frames-per-second (for video feeds)

        Returns:
            dict containing:
                - count: number of people detected
                - boxes: list of bounding boxes with coordinates and confidence
                - density_level: "NORMAL", "MODERATE", or "CRITICAL"
                - risk_badge: readable status badge string with emoji
                - occupancy_pct: percentage of capacity limit (if provided)
                - inference_time_ms: time taken by model forward pass
                - device_used: "cuda" or "cpu"
                - annotated_image: RGB image with visual overlays
        """
        start_time = time.perf_counter()

        # Run YOLO inference specifically filtering for class 0 (person)
        results = self.model(
            image_rgb,
            conf=confidence_threshold,
            classes=[0],
            verbose=False,
            device=self.device
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse detected bounding boxes
        boxes_data = []
        if len(results) > 0 and results[0].boxes is not None:
            raw_boxes = results[0].boxes
            for box in raw_boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu().numpy())
                boxes_data.append({
                    "bbox": [round(c, 1) for c in coords],
                    "confidence": round(conf, 3),
                })

        count = len(boxes_data)
        density_level, risk_badge, color_rgb = self.classify_density(count, capacity_limit)

        occupancy_pct = None
        if capacity_limit and capacity_limit > 0:
            occupancy_pct = round((count / capacity_limit) * 100.0, 1)

        # Draw visual annotations if requested
        if draw_annotations:
            annotated = self._draw_annotations(
                image_rgb=image_rgb.copy(),
                boxes=boxes_data,
                count=count,
                density_level=density_level,
                risk_badge=risk_badge,
                color_rgb=color_rgb,
                inference_time_ms=inference_time_ms,
                fps=fps,
                capacity_limit=capacity_limit,
                occupancy_pct=occupancy_pct
            )
        else:
            annotated = image_rgb

        return {
            "count": count,
            "boxes": boxes_data,
            "density_level": density_level,
            "risk_badge": risk_badge,
            "occupancy_pct": occupancy_pct,
            "inference_time_ms": round(inference_time_ms, 1),
            "device_used": self.device.upper(),
            "annotated_image": annotated,
        }

    def _draw_annotations(
        self,
        image_rgb: np.ndarray,
        boxes: List[Dict],
        count: int,
        density_level: str,
        risk_badge: str,
        color_rgb: Tuple[int, int, int],
        inference_time_ms: float,
        fps: Optional[float] = None,
        capacity_limit: Optional[int] = None,
        occupancy_pct: Optional[float] = None
    ) -> np.ndarray:
        """Renders bounding boxes and a Heads-Up Display (HUD) banner on the frame."""
        h, w = image_rgb.shape[:2]
        # Convert RGB color to BGR for OpenCV drawing functions
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # 1. Draw sleek bounding boxes for each person
        for item in boxes:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            conf = item["confidence"]

            # Box outline
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color_rgb, 2)

            # Tiny tag above head
            label = f"{int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                image_rgb,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 6, y1),
                color_rgb,
                -1
            )
            cv2.putText(
                image_rgb,
                label,
                (x1 + 3, max(th, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 2. Draw Top HUD Banner
        hud_height = 44
        hud_overlay = image_rgb.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_height), (20, 24, 30), -1)
        # Apply slight transparency
        cv2.addWeighted(hud_overlay, 0.85, image_rgb, 0.15, 0, image_rgb)

        # Left Accent status indicator stripe
        cv2.rectangle(image_rgb, (0, 0), (8, hud_height), color_rgb, -1)

        # HUD Text items
        hud_text = f"CrowdPulse | Count: {count} persons"
        if occupancy_pct is not None:
            hud_text += f" ({occupancy_pct}% cap: {capacity_limit})"
        hud_text += f" | {density_level}"

        perf_text = f"{self.device.upper()} | {inference_time_ms:.0f} ms"
        if fps is not None:
            perf_text += f" | {fps:.1f} FPS"

        # Draw left HUD text
        cv2.putText(
            image_rgb,
            hud_text,
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Draw right performance text
        (ptw, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            image_rgb,
            perf_text,
            (max(w - ptw - 16, 250), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 200, 220),
            1,
            cv2.LINE_AA
        )

        return image_rgb

