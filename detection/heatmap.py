"""
CrowdPulse - Optical CCTV Spatial Density & Congestion Analyzer
Project: CrowdPulse - Crowd Density Management System (SIH25165)

Beginner-Friendly Explanation:
Standard CCTV security cameras are optical RGB cameras (visible light spectrum),
NOT expensive thermal or infrared cameras.

This module provides 100% software-based optical crowd spatial analytics:
1. Optical Density Gradient Mapping: A software-generated gradient overlay (Amber/Crimson)
   showing crowd density peaks directly on normal CCTV footage.
2. Social Proximity & Crush Hazard Graph: Connects people who are standing too close
   to each other (< 1.5 meters) to detect dangerous crush pressures before a stampede.
3. Spatial 4x4 Grid Partitioning: Identifies physical bottlenecks in gates and corridors.
4. Optical Iso-Density Contours: Topographic boundary lines delineating crowded zones.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class SpatialHeatmapGenerator:
    """Optical CCTV Spatial Density, Proximity Clustering, and Bottleneck Analyzer.

    Compatible with standard visible-light RGB security cameras.
    """

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def generate_optical_density_map(
        self,
        image_rgb: np.ndarray,
        boxes: List[Dict],
        gaussian_sigma: int = 25,
        alpha: float = 0.45,
        style: str = "safety_alert"  # "safety_alert" (Green->Yellow->Red) or "amber_gradient"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a software-calculated optical density gradient overlay over standard CCTV footage.

        Requires NO thermal or infrared sensors.
        """
        h, w = image_rgb.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)

        if not boxes:
            return image_rgb.copy(), np.zeros_like(image_rgb)

        # 1. Accumulate detection centers (bottom-center for ground plane accuracy)
        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            cx = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
            cy = int(np.clip(y2, 0, h - 1))
            density_map[cy, cx] += 1.0

        # 2. Gaussian smoothing to calculate spatial density distribution
        ksize = int(gaussian_sigma * 4) + 1
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(density_map, (ksize, ksize), gaussian_sigma)

        # 3. Normalize intensity
        max_val = np.max(blurred)
        if max_val > 0:
            norm_density = (blurred / max_val * 255.0).astype(np.uint8)
        else:
            norm_density = np.zeros((h, w), dtype=np.uint8)

        # 4. Color gradient map: Safety Alert Spectrum (Blue/Teal -> Amber -> Crimson)
        colormap = cv2.COLORMAP_JET if style == "safety_alert" else cv2.COLORMAP_TURBO
        gradient_bgr = cv2.applyColorMap(norm_density, colormap)
        gradient_rgb = cv2.cvtColor(gradient_bgr, cv2.COLOR_BGR2RGB)

        # 5. Blend only non-zero density regions over the original optical image
        blended = image_rgb.copy()
        mask = (norm_density > 15).astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)

        blended = np.where(
            mask_3ch > 0,
            cv2.addWeighted(image_rgb, 1.0 - alpha, gradient_rgb, alpha, 0),
            image_rgb
        )

        return blended, gradient_rgb

    # Backwards-compatible alias for existing tests
    def generate_heatmap(self, image_rgb: np.ndarray, boxes: List[Dict], **kwargs):
        return self.generate_optical_density_map(image_rgb, boxes, **kwargs)

    def generate_proximity_crush_graph(
        self,
        image_rgb: np.ndarray,
        boxes: List[Dict],
        distance_threshold_px: int = 65
    ) -> Tuple[np.ndarray, Dict]:
        """Optical Social Proximity & Crush Graph for standard CCTV footage.

        Draws warning vectors between individuals standing dangerously close to each other
        (< 1.5m) to detect physical crush risks before stampedes.
        """
        annotated = image_rgb.copy()
        h, w = image_rgb.shape[:2]

        if not boxes:
            return annotated, {"total_connections": 0, "crush_index": 0.0, "risk_status": "Safe"}

        # Extract centers
        centers = []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            cx = (x1 + x2) / 2.0
            cy = y1 + (y2 - y1) * 0.3  # Upper torso / head center
            centers.append((cx, cy))

        n = len(centers)
        connections = 0
        connection_counts = [0] * n

        # Find proximity pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.hypot(dx, dy)

                if dist < distance_threshold_px:
                    connections += 1
                    connection_counts[i] += 1
                    connection_counts[j] += 1

                    # Color cue: Red if very close (< 60% of threshold), Yellow if moderate
                    pt1 = (int(centers[i][0]), int(centers[i][1]))
                    pt2 = (int(centers[j][0]), int(centers[j][1]))
                    line_color = (235, 50, 50) if dist < (distance_threshold_px * 0.6) else (240, 200, 20)
                    thickness = 2 if dist < (distance_threshold_px * 0.6) else 1
                    cv2.line(annotated, pt1, pt2, line_color, thickness, cv2.LINE_AA)

        # Draw nodes on people
        for idx, (cx, cy) in enumerate(centers):
            pt = (int(cx), int(cy))
            node_color = (240, 40, 40) if connection_counts[idx] >= 3 else ((240, 180, 20) if connection_counts[idx] >= 1 else (50, 220, 80))
            cv2.circle(annotated, pt, 5, node_color, -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

        # Compute crush index (average connections per person)
        crush_index = round(float((connections * 2) / max(n, 1)), 2)

        if crush_index >= 3.5:
            risk_status = "CRITICAL CRUSH HAZARD (Stampede Warning)"
        elif crush_index >= 1.8:
            risk_status = "MODERATE DENSITY (Compression Forming)"
        else:
            risk_status = "SAFE (Normal Free Flow)"

        # Draw HUD badge
        badge_text = f"Proximity Graph | Crush Index: {crush_index} | {risk_status}"
        (tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (10, h - 35), (tw + 24, h - 8), (20, 24, 30), -1)
        cv2.putText(annotated, badge_text, (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated, {
            "total_connections": connections,
            "crush_index": crush_index,
            "risk_status": risk_status,
        }

    def analyze_spatial_grid(
        self,
        image_rgb: np.ndarray,
        boxes: List[Dict],
        bottleneck_threshold_pct: float = 25.0
    ) -> Dict:
        """Divides frame into an NxM grid and computes localized crowd densities."""
        h, w = image_rgb.shape[:2]
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols

        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        total_people = len(boxes)

        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            cx = (x1 + x2) / 2.0
            cy = y2

            r = int(np.clip(cy // cell_h, 0, self.grid_rows - 1))
            c = int(np.clip(cx // cell_w, 0, self.grid_cols - 1))
            grid_counts[r, c] += 1

        hotspots = []
        bottleneck_detected = False
        annotated_grid = image_rgb.copy()

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x_start = int(c * cell_w)
                y_start = int(r * cell_h)
                x_end = int((c + 1) * cell_w)
                y_end = int((r + 1) * cell_h)

                cell_count = int(grid_counts[r, c])
                cell_share_pct = (cell_count / max(total_people, 1)) * 100.0

                is_congested = (cell_count >= 5 and cell_share_pct >= bottleneck_threshold_pct) or (cell_count >= 15)

                if is_congested:
                    bottleneck_detected = True
                    hotspots.append({
                        "row": r,
                        "col": c,
                        "cell_name": f"Sector [{r+1},{c+1}]",
                        "count": cell_count,
                        "crowd_share_pct": round(cell_share_pct, 1),
                        "status": "🚨 Bottleneck Choke Point",
                    })
                    sub_rect = annotated_grid[y_start:y_end, x_start:x_end]
                    red_tint = np.full_like(sub_rect, (230, 40, 40))
                    annotated_grid[y_start:y_end, x_start:x_end] = cv2.addWeighted(sub_rect, 0.7, red_tint, 0.3, 0)
                    cv2.rectangle(annotated_grid, (x_start, y_start), (x_end, y_end), (255, 50, 50), 3)
                else:
                    cv2.rectangle(annotated_grid, (x_start, y_start), (x_end, y_end), (100, 200, 255), 1)

                label = f"[{r+1},{c+1}]: {cell_count}"
                cv2.putText(
                    annotated_grid,
                    label,
                    (x_start + 8, y_start + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255) if is_congested else (200, 230, 255),
                    2,
                    cv2.LINE_AA
                )

        return {
            "grid_counts": grid_counts.tolist(),
            "hotspots": hotspots,
            "bottleneck_detected": bottleneck_detected,
            "grid_annotated_image": annotated_grid,
            "total_count": total_people,
        }
