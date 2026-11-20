"""
CrowdPulse - Real CCTV Real-Time Video Benchmark & Test Runner
Tests standard optical surveillance CCTV footage through the entire pipeline:
- YOLOv8 Person Detection
- Optical Density Gradient Mapping
- Social Proximity & Crush Graph (< 1.5m)
- Multi-Agent Coordination (Monitor -> Anomaly -> Dispatch)
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from detection.detector import CrowdDetector
from detection.heatmap import SpatialHeatmapGenerator
from agents.coordinator import MultiAgentCoordinator

def run_real_cctv_test(video_filename="real_cctv_pedestrians.mp4", max_frames=90):
    video_path = PROJECT_ROOT / "data" / "videos" / video_filename
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at {video_path}")

    out_dir = PROJECT_ROOT / "data" / "cctv_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" CROWDPULSE - REAL-WORLD CCTV STREAMING BENCHMARK")
    print(f" Source Video: {video_path.name}")
    print(f" Processing Target: Up to {max_frames} frames on GPU/CPU")
    print("=" * 70)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video Info: {vid_width}x{vid_height} @ {vid_fps:.1f} FPS, {total_frames} total frames")

    # Initialize components
    detector = CrowdDetector()
    spatial_engine = SpatialHeatmapGenerator(grid_rows=4, grid_cols=4)
    coordinator = MultiAgentCoordinator()

    # Video writer for annotated output
    annotated_video_path = out_dir / "annotated_cctv_stream.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_video_path), fourcc, int(vid_fps), (vid_width, vid_height))

    frame_idx = 0
    counts = []
    latencies = []
    fps_list = []
    agent_anomalies = []
    agent_dispatches = []
    sample_annotated_frame = None
    sample_density_frame = None
    sample_crush_frame = None

    start_stream_time = time.perf_counter()

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.perf_counter()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Optical Detection
        det_res = detector.detect_people(
            image_rgb=frame_rgb,
            confidence_threshold=0.20,
            capacity_limit=20,
            mode="Standard Optical (YOLO)",
            draw_annotations=True
        )

        boxes = det_res["boxes"]
        count = det_res["count"]
        counts.append(count)

        # 2. Optical Spatial Density Gradient
        density_blended_rgb, density_pure_rgb = spatial_engine.generate_optical_density_map(
            image_rgb=frame_rgb,
            boxes=boxes,
            alpha=0.55
        )

        # 3. Social Proximity & Crush Graph (< 1.5m)
        crush_annotated_rgb, crush_metrics = spatial_engine.generate_proximity_crush_graph(
            image_rgb=frame_rgb,
            boxes=boxes,
            distance_threshold_px=65
        )

        # 4. 4x4 Choke Point Spatial Grid
        grid_res = spatial_engine.analyze_spatial_grid(
            image_rgb=frame_rgb,
            boxes=boxes
        )

        # 5. Autonomous Multi-Agent Cycle (Monitor -> Anomaly -> Dispatch)
        agent_out = coordinator.process_zone_update(
            zone_id="ZONE-CCTV-01",
            current_count=count,
            capacity_limit=20,
            bottleneck_detected=grid_res["bottleneck_detected"],
            hotspot_count=len(grid_res["hotspots"]),
            inference_latency_ms=det_res["inference_time_ms"],
        )

        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed * 1000.0)
        instant_fps = 1.0 / max(t_elapsed, 1e-5)
        fps_list.append(instant_fps)

        if agent_out["anomaly"]:
            agent_anomalies.append(agent_out["anomaly"])
        if agent_out["actions"]:
            agent_dispatches.extend(agent_out["actions"])

        # Write frame to video
        frame_out_bgr = cv2.cvtColor(det_res["annotated_image"], cv2.COLOR_RGB2BGR)
        writer.write(frame_out_bgr)

        # Save mid-stream sample images for reporting
        if frame_idx == 45 or (frame_idx == 1 and sample_annotated_frame is None):
            sample_annotated_frame = det_res["annotated_image"]
            sample_density_frame = density_blended_rgb
            sample_crush_frame = crush_annotated_rgb

        if frame_idx % 20 == 0 or frame_idx == max_frames:
            print(f"  [Frame {frame_idx:02d}/{max_frames}] Count: {count:2d} people | Latency: {t_elapsed*1000:.1f}ms | FPS: {instant_fps:.1f} | Risk: {det_res['density_level']}")

    cap.release()
    writer.release()
    total_stream_time = time.perf_counter() - start_stream_time

    # Save snapshot images
    if sample_annotated_frame is not None:
        cv2.imwrite(str(out_dir / "cctv_frame_detected.jpg"), cv2.cvtColor(sample_annotated_frame, cv2.COLOR_RGB2BGR))
    if sample_density_frame is not None:
        cv2.imwrite(str(out_dir / "cctv_frame_density_map.jpg"), cv2.cvtColor(sample_density_frame, cv2.COLOR_RGB2BGR))
    if sample_crush_frame is not None:
        cv2.imwrite(str(out_dir / "cctv_frame_crush_graph.jpg"), cv2.cvtColor(sample_crush_frame, cv2.COLOR_RGB2BGR))

    # Summary Statistics
    avg_count = float(np.mean(counts)) if counts else 0.0
    peak_count = int(np.max(counts)) if counts else 0
    min_count = int(np.min(counts)) if counts else 0
    avg_fps = float(np.mean(fps_list)) if fps_list else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 70)
    print(" CCTV REAL-TIME BENCHMARK RESULTS")
    print("=" * 70)
    print(f" Total Frames Processed:   {frame_idx} frames in {total_stream_time:.2f}s")
    print(f" Processing Speed:         {avg_fps:.1f} FPS (Mean Latency: {avg_latency:.1f} ms/frame)")
    print(f" Headcount Range:          Min: {min_count} | Mean: {avg_count:.1f} | Peak: {peak_count} persons")
    print(f" Agent Anomalies Flagged:  {len(agent_anomalies)} events")
    print(f" Agent Dispatch Actions:   {len(agent_dispatches)} orders issued")
    print(f" Annotated Video Saved:    {annotated_video_path}")
    print(f" Snapshot Overlays Saved:  {out_dir}")
    print("=" * 70)

    return {
        "frames_processed": frame_idx,
        "avg_fps": round(avg_fps, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "min_count": min_count,
        "max_count": peak_count,
        "mean_count": round(avg_count, 1),
        "anomalies_count": len(agent_anomalies),
        "dispatches_count": len(agent_dispatches),
        "annotated_video": str(annotated_video_path),
        "snapshots_dir": str(out_dir),
    }

if __name__ == "__main__":
    run_real_cctv_test()
