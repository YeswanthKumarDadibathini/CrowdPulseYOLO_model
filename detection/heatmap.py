"""
CrowdPulse - Spatial Grid & Density Heatmap Generator (Phase 3)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 3 Spatial Analytics & Incident Database

Beginner-Friendly Explanation:
In crowd safety management, knowing the total number of people is not enough.
We must know WHERE the crowd is gathering.

If 100 people are spread evenly across a large plaza, it is completely safe.
If those same 100 people rush toward a single narrow exit gate, it creates a dangerous
stampede bottleneck!

This module provides two spatial analysis tools:
1. Spatial Grid Partitioning: Divides the camera view into rows & columns (e.g. 4x4)
   and counts people in each grid block to highlight congested cells.
2. 2D Gaussian Density Heatmap: Generates a thermal colored overlay (Blue -> Yellow -> Red)
   showing intense clustering hotspots.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class SpatialHeatmapGenerator:
    """Generates continuous Gaussian density heatmaps and spatial grid hotspot analyses."""

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4):
        """Initializes grid configuration."""
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def generate_heatmap(
        self,
        image_rgb: np.ndarray,
        boxes: List[Dict],
        gaussian_sigma: int = 25,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a thermal density heatmap blended over the original image.

        Parameters:
            image_rgb: Original RGB image
            boxes: List of bounding box dicts from CrowdDetector (each with 'bbox': [x1, y1, x2, y2])
            gaussian_sigma: Blur radius of crowd points (higher = broader heat spots)
            colormap: OpenCV colormap (default COLORMAP_JET: Blue = Low, Red = High)
            alpha: Transparency of the heatmap overlay (0.0 to 1.0)

        Returns:
            tuple: (blended_image_rgb, raw_heatmap_rgb)
        """
        h, w = image_rgb.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)

        if not boxes:
            # If no people detected, return original image
            return image_rgb.copy(), np.zeros_like(image_rgb)

        # 1. Accumulate detection centers
        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            # We use bottom-center (feet position) which accurately places a person on the ground plane
            cx = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
            cy = int(np.clip(y2, 0, h - 1))
            density_map[cy, cx] += 1.0

        # 2. Smooth discrete points with a Gaussian kernel
        ksize = int(gaussian_sigma * 4) + 1
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(density_map, (ksize, ksize), gaussian_sigma)

        # 3. Normalize intensity to 0 - 255
        max_val = np.max(blurred)
        if max_val > 0:
            norm_density = (blurred / max_val * 255.0).astype(np.uint8)
        else:
            norm_density = np.zeros((h, w), dtype=np.uint8)

        # 4. Apply colormap
        heatmap_bgr = cv2.applyColorMap(norm_density, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # 5. Blend only non-zero density regions with the original image
        blended = image_rgb.copy()
        # Create a mask where density is present (ignore cold/empty regions)
        mask = (norm_density > 15).astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)

        # Alpha blend where mask is active
        blended = np.where(
            mask_3ch > 0,
            cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0),
            image_rgb
        )

        return blended, heatmap_rgb

    def analyze_spatial_grid(
        self,
        image_rgb: np.ndarray,
        boxes: List[Dict],
        bottleneck_threshold_pct: float = 25.0
    ) -> Dict:
        """Divides frame into an NxM grid and computes localized crowd densities.

        Parameters:
            image_rgb: Input image
            boxes: Detected bounding boxes
            bottleneck_threshold_pct: Percentage of total crowd in a single cell that constitutes a bottleneck

        Returns:
            dict containing:
                - grid_counts: 2D matrix (rows x cols) of counts
                - hotspots: list of congested grid cells
                - bottleneck_detected: bool
                - grid_annotated_image: image with drawn grid overlays and cell labels
        """
        h, w = image_rgb.shape[:2]
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols

        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        total_people = len(boxes)

        # Assign each person to a grid cell
        for item in boxes:
            x1, y1, x2, y2 = item["bbox"]
            cx = (x1 + x2) / 2.0
            cy = y2  # Feet position

            r = int(np.clip(cy // cell_h, 0, self.grid_rows - 1))
            c = int(np.clip(cx // cell_w, 0, self.grid_cols - 1))
            grid_counts[r, c] += 1

        # Check for bottleneck cells
        hotspots = []
        bottleneck_detected = False

        annotated_grid = image_rgb.copy()

        # Draw grid lines and counts
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
                        "cell_name": f"Grid Cell [{r+1},{c+1}]",
                        "count": cell_count,
                        "crowd_share_pct": round(cell_share_pct, 1),
                        "status": "🚨 Bottleneck Hotspot",
                    })
                    # Highlight danger cell with red overlay
                    sub_rect = annotated_grid[y_start:y_end, x_start:x_end]
                    red_tint = np.full_like(sub_rect, (230, 40, 40))
                    annotated_grid[y_start:y_end, x_start:x_end] = cv2.addWeighted(sub_rect, 0.7, red_tint, 0.3, 0)
                    cv2.rectangle(annotated_grid, (x_start, y_start), (x_end, y_end), (255, 50, 50), 3)
                else:
                    # Normal grid cell boundary
                    cv2.rectangle(annotated_grid, (x_start, y_start), (x_end, y_end), (100, 200, 255), 1)

                # Draw cell label (e.g. "[1,1]: 4")
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

