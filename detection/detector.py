"""
CrowdPulse - Hybrid Optical CCTV Crowd Detection & Deep Headcount Engine
Project: CrowdPulse - Crowd Density Management System (SIH25165)

Beginner-Friendly Explanation:
This module combines TWO complementary pretrained deep learning models to solve
both spatial localization AND accurate dense crowd headcounting on standard
OPTICAL RGB CCTV cameras (NO thermal or infrared hardware required):

1. Pretrained Object Detector (YOLOv8s / YOLOv8n):
   Locates individual person bounding boxes [x1, y1, x2, y2] for the Social Proximity
   Crush Graph, 4x4 Choke-Point Grid, and live CCTV HUD overlays.
2. Fine-Tuned Deep Crowd Counter (CrowdPulseCountNet — ConvNeXt-Tiny / ResNet-50):
   A pretrained ImageNet CNN backbone fine-tuned directly on your 1,198-image
   Dataset_CrowdPulse benchmark to predict accurate headcounts even in massive
   crowds (100 to 3,000+ people) where bounding boxes overlap.
"""

import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure Ultralytics config is placed safely inside the project workspace
_workspace_yolo_dir = Path(__file__).resolve().parent.parent / ".yolo_config"
_workspace_yolo_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_workspace_yolo_dir)

import torch
from ultralytics import YOLO

import config
from detection.crowd_counter_model import CrowdCountPredictor


class CrowdDetector:
    """Hybrid Optical CCTV Crowd Density and Person Detection Engine."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        count_checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """Initializes YOLO object detector and optional fine-tuned deep crowd counter."""
        if model_path is None:
            # Prefer higher-accuracy yolov8s.pt if available, else yolov8n.pt
            candidates = [
                config.MODELS_DIR / "yolov8s.pt",
                config.MODELS_DIR / "yolov8n.pt",
                config.BASE_DIR / "yolov8n.pt",
            ]
            self.model_path = next((p for p in candidates if p.exists()), candidates[-1])
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights not found at {self.model_path}. "
                f"Ensure 'yolov8s.pt' or 'yolov8n.pt' is present in the models/ directory."
            )

        # Device selection: automatically uses NVIDIA CUDA GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        # Warm up YOLO once
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, classes=[0], verbose=False)

        # Load fine-tuned CrowdPulseCountNet if checkpoint exists
        self.count_predictor = CrowdCountPredictor(checkpoint_path=count_checkpoint_path)

    @property
    def has_trained_counter(self) -> bool:
        """Returns True if the custom fine-tuned deep crowd counter is loaded."""
        return self.count_predictor.is_ready

    @property
    def yolo_model_name(self) -> str:
        return self.model_path.name

    def classify_density(
        self,
        count: int,
        capacity_limit: Optional[int] = None
    ) -> Tuple[str, str, Tuple[int, int, int]]:
        """Determines the risk classification based on people count and capacity limit."""
        if capacity_limit and capacity_limit > 0:
            occupancy_pct = (count / capacity_limit) * 100.0
            if occupancy_pct < config.DENSITY_THRESHOLDS["NORMAL"]:
                return "NORMAL", "🟢 Safe", (46, 204, 113)
            elif occupancy_pct < config.DENSITY_THRESHOLDS["MODERATE"]:
                return "MODERATE", "🟡 Caution", (241, 196, 15)
            else:
                return "CRITICAL", "🔴 Overcrowded", (231, 76, 60)
        else:
            if count <= 15:
                return "NORMAL", "🟢 Low Density", (46, 204, 113)
            elif count <= 40:
                return "MODERATE", "🟡 Medium Density", (241, 196, 15)
            else:
                return "CRITICAL", "🔴 High Density", (231, 76, 60)

    def _detect_single_pass(
        self,
        image_rgb: np.ndarray,
        confidence_threshold: float,
        imgsz: int = 640
    ) -> List[Dict]:
        """Runs a single YOLO forward pass with adaptive resolution."""
        results = self.model(
            image_rgb,
            conf=confidence_threshold,
            classes=[0],
            imgsz=imgsz,
            verbose=False,
            device=self.device
        )
        boxes_data = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                boxes_data.append({
                    "bbox": [round(c, 1) for c in coords],
                    "confidence": round(conf, 3),
                })
        return boxes_data

    def _detect_sliced(
        self,
        image_rgb: np.ndarray,
        confidence_threshold: float,
        tile_size: int = 480,
        overlap: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """Runs high-density sliced YOLO inference by splitting frame into overlapping tiles."""
        h, w = image_rgb.shape[:2]
        all_boxes = []

        # 1. Full-image high-resolution pass
        full_boxes = self._detect_single_pass(
            image_rgb,
            confidence_threshold,
            imgsz=max(640, min(w, 1024)),
        )
        all_boxes.extend(full_boxes)

        # 2. Overlapping tile passes for small / distant people
        step_y = int(tile_size * (1 - overlap))
        step_x = int(tile_size * (1 - overlap))

        for y in range(0, max(1, h - tile_size + step_y), step_y):
            for x in range(0, max(1, w - tile_size + step_x), step_x):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tile = image_rgb[y:y_end, x:x_end]

                tile_results = self._detect_single_pass(tile, confidence_threshold, imgsz=tile_size)
                for item in tile_results:
                    bx1, by1, bx2, by2 = item["bbox"]
                    conf = item["confidence"]
                    all_boxes.append({
                        "bbox": [bx1 + x, by1 + y, bx2 + x, by2 + y],
                        "confidence": conf
                    })

        if not all_boxes:
            return []

        # 3. Merge overlapping boxes using NMS
        xywh = []
        scores = []
        for b in all_boxes:
            bx1, by1, bx2, by2 = b["bbox"]
            xywh.append([int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)])
            scores.append(float(b["confidence"]))

        indices = cv2.dnn.NMSBoxes(
            xywh,
            scores,
            score_threshold=confidence_threshold,
            nms_threshold=iou_threshold,
        )
        if len(indices) == 0:
            return []

        merged_boxes = []
        for idx in indices:
            i = int(idx[0]) if isinstance(idx, (list, np.ndarray)) else int(idx)
            merged_boxes.append(all_boxes[i])

        return merged_boxes

    def _estimate_dense_crowd_calibration(
        self,
        image_rgb: np.ndarray,
        detected_count: int,
        boxes: List[Dict]
    ) -> int:
        """Fallback optical crowd density estimator when trained CNN checkpoint is absent."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        edge_energy = np.abs(lap)
        blur_energy = cv2.GaussianBlur(edge_energy, (31, 31), 10)
        crowd_mask = blur_energy > np.mean(blur_energy)

        crowd_coverage_ratio = np.sum(crowd_mask) / max(h * w, 1)
        if crowd_coverage_ratio > 0.15:
            local_max = cv2.dilate(gray, np.ones((5, 5), np.uint8))
            peaks = (gray == local_max) & crowd_mask & (gray > 40)
            peak_count = int(np.sum(peaks))
            return int(max(detected_count, (detected_count * 0.4) + (peak_count * 0.6)))
        return detected_count

    def detect_people(
        self,
        image_rgb: np.ndarray,
        confidence_threshold: float = 0.15,
        capacity_limit: Optional[int] = None,
        draw_annotations: bool = True,
        mode: str = "standard",  # "trained_hybrid", "high_density_sliced", "calibrated", "standard"
        fps: Optional[float] = None
    ) -> Dict:
        """Runs Optical CCTV crowd detection and headcounting.

        Modes:
            - "trained_hybrid": Uses fine-tuned CrowdPulseCountNet (ConvNeXt-Tiny / ResNet-50)
                                for accurate dense headcount + YOLOv8 for person bounding boxes.
            - "high_density_sliced": Multi-tile sliced YOLO inference for maximum box recall.
            - "calibrated": Uses trained neural model if loaded, else optical density fallback.
            - "standard": Fast single-pass YOLO for real-time video streams.
        """
        start_time = time.perf_counter()
        neural_info = None

        if mode == "trained_hybrid":
            # Fast YOLO pass for spatial boxes & proximity crush graph
            boxes_data = self._detect_single_pass(
                image_rgb,
                confidence_threshold=confidence_threshold,
                imgsz=max(640, min(image_rgb.shape[1], 1024)),
            )
            detected_count = len(boxes_data)

            if self.has_trained_counter:
                neural_info = self.count_predictor.predict(image_rgb, use_tta=True)
                final_count = neural_info["predicted_count"]
            else:
                final_count = detected_count

        elif mode == "high_density_sliced":
            boxes_data = self._detect_sliced(image_rgb, confidence_threshold=confidence_threshold)
            detected_count = len(boxes_data)
            final_count = detected_count

        elif mode == "calibrated":
            boxes_data = self._detect_sliced(image_rgb, confidence_threshold=confidence_threshold)
            detected_count = len(boxes_data)
            if self.has_trained_counter:
                neural_info = self.count_predictor.predict(image_rgb, use_tta=True)
                final_count = neural_info["predicted_count"]
            else:
                final_count = self._estimate_dense_crowd_calibration(image_rgb, detected_count, boxes_data)

        else:
            boxes_data = self._detect_single_pass(image_rgb, confidence_threshold=confidence_threshold)
            detected_count = len(boxes_data)
            final_count = detected_count

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        density_level, risk_badge, color_rgb = self.classify_density(final_count, capacity_limit)

        occupancy_pct = None
        if capacity_limit and capacity_limit > 0:
            occupancy_pct = round((final_count / capacity_limit) * 100.0, 1)

        if draw_annotations:
            annotated = self._draw_annotations(
                image_rgb=image_rgb.copy(),
                boxes=boxes_data,
                count=final_count,
                density_level=density_level,
                risk_badge=risk_badge,
                color_rgb=color_rgb,
                inference_time_ms=inference_time_ms,
                fps=fps,
                capacity_limit=capacity_limit,
                occupancy_pct=occupancy_pct,
                mode=mode,
                neural_backbone=neural_info["backbone"] if neural_info else None,
            )
        else:
            annotated = image_rgb

        return {
            "count": final_count,
            "detected_boxes_count": detected_count,
            "boxes": boxes_data,
            "density_level": density_level,
            "risk_badge": risk_badge,
            "occupancy_pct": occupancy_pct,
            "inference_time_ms": round(inference_time_ms, 1),
            "device_used": self.device.upper(),
            "mode_used": mode,
            "neural_prediction": neural_info,
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
        occupancy_pct: Optional[float] = None,
        mode: str = "standard",
        neural_backbone: Optional[str] = None,
    ) -> np.ndarray:
        """Renders bounding boxes and an Optical CCTV Heads-Up Display (HUD) banner."""
        h, w = image_rgb.shape[:2]

        for item in boxes:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            conf = item["confidence"]

            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color_rgb, 2)

            label = f"{int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(
                image_rgb,
                (x1, max(0, y1 - th - 5)),
                (x1 + tw + 4, y1),
                color_rgb,
                -1
            )
            cv2.putText(
                image_rgb,
                label,
                (x1 + 2, max(th, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        hud_height = 44
        hud_overlay = image_rgb.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_height), (18, 22, 28), -1)
        cv2.addWeighted(hud_overlay, 0.85, image_rgb, 0.15, 0, image_rgb)
        cv2.rectangle(image_rgb, (0, 0), (8, hud_height), color_rgb, -1)

        if neural_backbone:
            mode_tag = f" [AI:{neural_backbone.upper()}]"
        elif mode == "high_density_sliced":
            mode_tag = " [SLICED HQ]"
        elif mode == "calibrated":
            mode_tag = " [CALIBRATED]"
        else:
            mode_tag = ""

        hud_text = f"CrowdPulse Optical | Count: {count}{mode_tag}"
        if occupancy_pct is not None:
            hud_text += f" ({occupancy_pct}% of {capacity_limit})"
        hud_text += f" | {density_level}"

        perf_text = f"{self.device.upper()} | {inference_time_ms:.0f} ms"
        if fps is not None:
            perf_text += f" | {fps:.1f} FPS"

        cv2.putText(image_rgb, hud_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

        (ptw, _), _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(image_rgb, perf_text, (max(w - ptw - 16, 250), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 205, 230), 1, cv2.LINE_AA)

        return image_rgb
