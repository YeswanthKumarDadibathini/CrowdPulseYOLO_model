"""
CrowdPulse - Optical CCTV Multi-Agent Web Application
Project: CrowdPulse - Crowd Density Management System (SIH25165)

Beginner-Friendly Explanation:
This Streamlit application provides an interactive web dashboard for standard
OPTICAL RGB CCTV cameras (NO thermal/infrared hardware required):
1. Multi-Zone Dashboard: Overview of 4 monitored zones with real-time risk levels.
2. Multi-Agent Command Center: Live coordination of Monitor, Anomaly, and Dispatch Agents.
3. Optical Spatial Density & Crush Graph: Social proximity clustering, 4x4 grid choke-points,
   and optical density mapping.
4. Dataset Benchmarking: Test Enhanced Sliced & Calibrated YOLO on the 1,198 crowd images.
5. Custom Image Upload: Upload any photo for accurate headcount & agent evaluation.
6. CCTV Video Stream: Process live video feeds with bounding boxes, HUD, and FPS monitoring.
7. Incident Logs & Analytics: SQLite database logs, time-series charts, and alert management.
8. Hardware Diagnostics: View Python, PyTorch, CUDA, and GPU hardware status.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

import config
from agents.coordinator import MultiAgentCoordinator
from database.db_service import (
    get_active_alerts,
    get_recent_logs_dataframe,
    init_db,
    log_density_reading,
    seed_mock_history_if_empty,
    update_alert_status,
)
from detection.dataset_loader import (
    get_available_datasets,
    get_dataset_metadata,
    get_dataset_summary,
    load_dataset_sample,
)
from detection.detector import CrowdDetector
from detection.heatmap import SpatialHeatmapGenerator
from detection.video_stream import VideoStreamProcessor, generate_synthetic_demo_video
from services.dispatch_service import (
    DigitalSignageService,
    MarshalSquadTracker,
    TurnstileGateController,
)
from utils.environment_check import get_environment_report

# ==============================================================================
# Page Configuration & Database Initialization
# ==============================================================================
st.set_page_config(
    page_title=f"{config.PROJECT_NAME} - SIH25165 (Optical CCTV Multi-Agent)",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
seed_mock_history_if_empty(num_hours=4)


# ==============================================================================
# Cache Resources
# ==============================================================================
@st.cache_resource(show_spinner="Loading Optical YOLOv8 Model...")
def load_detector():
    return CrowdDetector()


@st.cache_resource
def load_spatial_analyzer():
    return SpatialHeatmapGenerator(grid_rows=4, grid_cols=4)


@st.cache_resource
def load_coordinator():
    return MultiAgentCoordinator()


@st.cache_resource
def load_dispatch_services():
    return (
        MarshalSquadTracker(),
        TurnstileGateController(),
        DigitalSignageService(),
    )


detector = load_detector()
spatial_analyzer = load_spatial_analyzer()
coordinator = load_coordinator()
squad_tracker, gate_controller, signage_service = load_dispatch_services()

active_alerts = get_active_alerts()

# Helper dictionary mapping UI labels to detector modes
DETECTION_MODES = {
    "🎯 High-Density Sliced Mode (Best Headcount Recall)": "high_density_sliced",
    "📊 Calibrated Crowd Estimator (Ultra-Dense Scenes)": "calibrated",
    "⚡ Standard Optical Mode (Fastest Real-Time)": "standard",
}

# ==============================================================================
# Sidebar Navigation
# ==============================================================================
with st.sidebar:
    st.title(f"👥 {config.PROJECT_NAME}")
    st.caption(f"Project ID: **{config.SIH_PROJECT_CODE}**")
    st.caption("📷 **Standard Optical RGB CCTV System**")
    st.divider()

    alert_badge_text = f" ({len(active_alerts)} 🚨)" if active_alerts else ""

    menu = st.radio(
        "Navigation",
        [
            "📍 Multi-Zone Overview",
            "🤖 Multi-Agent Command Center",
            "🗺️ Optical Spatial Density & Crush Graph",
            "🖼️ Crowd Dataset Benchmark",
            "📤 Upload Custom Image",
            "🎥 CCTV Video Stream",
            f"📈 Incident Logs & Analytics{alert_badge_text}",
            "🖥️ Hardware & Environment",
        ],
        index=0,
    )

    st.divider()
    st.subheader("System Status")
    device_badge = "🚀 CUDA (GPU)" if detector.device == "cuda" else "⚙️ CPU Mode"
    st.write("- **Camera Spec:** Standard RGB Optical")
    st.write(f"- **Engine:** Enhanced YOLOv8n (`person`)")
    st.write(f"- **Active Device:** `{device_badge}`")
    st.write(f"- **Active Incidents:** `{len(active_alerts)}`")

    st.divider()
    st.caption("CrowdPulse SIH25165 | B.Tech Final Year")


# ==============================================================================
# VIEW 1: Multi-Zone Overview
# ==============================================================================
if menu == "📍 Multi-Zone Overview":
    st.title(f"👥 {config.PROJECT_NAME} – Multi-Zone Overview")
    st.markdown(f"**SIH Problem Statement:** `{config.SIH_PROJECT_CODE}` | *{config.PROJECT_TAGLINE}*")

    st.info(
        "📷 **100% Standard Optical CCTV Compatible:** Works with ordinary security cameras and webcams—no "
        "expensive thermal or infrared hardware required. Autonomous Multi-Agent system is active."
    )

    if active_alerts:
        st.warning(f"⚠️ **{len(active_alerts)} Active Crowd Incidents** currently require attention! Check the 'Incident Logs & Analytics' tab.")

    st.subheader("📍 Live Monitored Zones")
    cols = st.columns(4)

    for idx, zone in enumerate(config.DEFAULT_ZONES):
        with cols[idx]:
            occ = (zone["current_count"] / zone["capacity_limit"]) * 100
            badge = "🟢" if zone["risk_level"] == "Low" else ("🟡" if zone["risk_level"] == "Medium" else "🔴")

            with st.container(border=True):
                st.markdown(f"### {badge} {zone['name']}")
                st.caption(f"Feed: `{zone['camera_source']}`")
                st.metric(
                    label="Current People Count",
                    value=f"{zone['current_count']} persons",
                    delta=f"{occ:.1f}% capacity",
                    delta_color="inverse" if occ > 80 else "normal",
                )
                st.write(f"**Safe Capacity:** `{zone['capacity_limit']} persons`")
                st.write(f"**Risk Level:** `{zone['risk_level']}`")
                st.write(f"**Status:** `{zone['demo_status']}`")

    st.divider()
    st.subheader("📊 Zone Occupancy Comparison")
    zone_df = pd.DataFrame([
        {
            "Zone": z["name"],
            "Current Count": z["current_count"],
            "Capacity Limit": z["capacity_limit"],
            "Occupancy (%)": round((z["current_count"] / z["capacity_limit"]) * 100, 1),
            "Risk": z["risk_level"],
        }
        for z in config.DEFAULT_ZONES
    ])

    fig = px.bar(
        zone_df,
        x="Zone",
        y="Occupancy (%)",
        color="Risk",
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"},
        text="Occupancy (%)",
        title="Real-Time Zone Occupancy vs Thresholds",
    )
    fig.add_hline(y=config.DENSITY_THRESHOLDS["MODERATE"], line_dash="dash", line_color="orange", annotation_text="Caution (80%)")
    fig.add_hline(y=config.DENSITY_THRESHOLDS["NORMAL"], line_dash="dot", line_color="green", annotation_text="Normal (60%)")
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# VIEW 2: Multi-Agent Command Center
# ==============================================================================
elif menu == "🤖 Multi-Agent Command Center":
    st.title("🤖 Multi-Agent Command Center")
    st.markdown(
        "Observe how **three specialized AI agents** collaborate autonomously in real time: "
        "**Monitor Agent** measures telemetry $\\to$ **Anomaly Agent** detects stampede/surge threats $\\to$ **Dispatch Agent** issues tactical orders."
    )

    ag_col1, ag_col2, ag_col3 = st.columns(3)
    with ag_col1:
        with st.container(border=True):
            st.markdown("### 👁️ Monitor Agent")
            st.caption("Role: Perception & Influx Velocity")
            st.write("- **Status:** 🟢 Active Polling")
            st.write("- **Metrics:** Headcount, $\\Delta N / \\Delta t$ (Surge Rate)")
            st.write("- **Output:** `ZoneTelemetry` payloads")
    with ag_col2:
        with st.container(border=True):
            st.markdown("### 🧠 Anomaly Agent")
            st.caption("Role: Threat & Risk Classification")
            st.write("- **Status:** 🟢 Real-time Scanning")
            st.write("- **Rules:** Surges, Choke Points, Overcrowding")
            st.write("- **Output:** `AnomalyReport` warnings")
    with ag_col3:
        with st.container(border=True):
            st.markdown("### ⚡ Dispatch Agent")
            st.caption("Role: Tactical Response & Control")
            st.write("- **Status:** 🟢 Standby / Ready")
            st.write("- **Actions:** Security squads, Turnstiles, PA")
            st.write("- **Output:** `DispatchAction` orders")

    st.divider()

    st.subheader("⚡ Live Incident Simulation & Agent Reaction")
    st.caption("Trigger synthetic crowd events to evaluate how the multi-agent system reacts in real time.")

    sc1, sc2, sc3 = st.columns([2, 2, 1])
    with sc1:
        scenario = st.selectbox(
            "Select Crowd Event Scenario:",
            [
                "Sudden Crowd Surge (Rapid Influx)",
                "Physical Bottleneck Cluster (Exit Choke Point)",
                "Dangerous Overcrowding (Capacity Breached)",
                "Normal Safe Operations",
            ]
        )
    with sc2:
        target_zone = st.selectbox(
            "Target Monitored Zone:",
            [f"{z['name']} ({z['id']})" for z in config.DEFAULT_ZONES]
        )

    sc_code = "SURGE"
    if "Bottleneck" in scenario:
        sc_code = "BOTTLENECK"
    elif "Overcrowding" in scenario:
        sc_code = "CRITICAL_OVERCROWDING"
    elif "Normal" in scenario:
        sc_code = "NORMAL"

    sel_zid = target_zone.split("(")[-1].replace(")", "").strip()

    with sc3:
        st.write("")
        trigger_btn = st.button("🚀 Trigger Scenario", type="primary")

    if trigger_btn:
        with st.spinner("Executing Multi-Agent lifecycle..."):
            sim_result = coordinator.simulate_crowd_scenario(scenario=sc_code, zone_id=sel_zid)

        st.success(f"✅ **Scenario Executed:** {sim_result['scenario_description']}")

        t_telemetry, t_anomaly, t_dispatch = st.tabs(["1️⃣ Monitor Agent Telemetry", "2️⃣ Anomaly Agent Findings", "3️⃣ Dispatch Action Orders"])

        with t_telemetry:
            tel = sim_result["telemetry"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Count", f"{tel['current_count']} persons")
            m2.metric("Occupancy", f"{tel['occupancy_pct']}%", f"Cap: {tel['capacity_limit']}")
            m3.metric("Surge Velocity", f"{tel['surge_rate_per_min']:+0.1f} / min")
            m4.metric("Bottleneck Choke", "🚨 Detected" if tel['bottleneck_detected'] else "None")
            st.json(tel)

        with t_anomaly:
            anom = sim_result["anomaly"]
            if anom:
                st.error(f"🚨 **{anom['severity']} THREAT DETECTED:** {anom['anomaly_type']}")
                st.write(f"**Confidence Score:** `{int(anom['confidence'] * 100)}%`")
                st.write(f"**Description:** {anom['description']}")
                st.json(anom)
            else:
                st.success("✅ **No Anomalies Detected:** Telemetry falls safely within normal operating thresholds.")

        with t_dispatch:
            actions = sim_result["actions"]
            if actions:
                st.warning(f"⚡ **{len(actions)} Tactical Action Items Issued:**")
                for act in actions:
                    with st.container(border=True):
                        st.markdown(f"**Order ID:** `{act['action_id']}` | **Target:** `{act['target_team']}` | **Priority:** `{act['priority']}`")
                        st.write(f"📋 **Instructions:** {act['instructions']}")
            else:
                st.info("ℹ️ No emergency dispatch required for normal crowd conditions.")

    st.divider()

    tab_bus, tab_field = st.tabs(["💬 Inter-Agent Message Bus Feed", "🛡️ Field Equipment & Squad Tracker"])

    with tab_bus:
        st.subheader("Central Event Bus Log")
        recent_msgs = coordinator.get_recent_messages(limit=15)
        for msg in recent_msgs:
            sender_emoji = "👁️" if "Monitor" in msg["sender"] else ("🧠" if "Anomaly" in msg["sender"] else "⚡")
            with st.chat_message(name=msg["sender"], avatar=sender_emoji):
                st.markdown(f"**[{msg['timestamp']}] {msg['sender']} $\\to$ `{msg['recipient']}`** | Type: `{msg['message_type']}`")
                st.json(msg["content"])

    with tab_field:
        st.subheader("Field Asset Deployment Status")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("#### 🛡️ On-Ground Security & Marshal Squads")
            squad_df = pd.DataFrame(squad_tracker.get_all_squads())
            st.dataframe(squad_df, hide_index=True)

        with f_col2:
            st.markdown("#### 🚧 Turnstile Gates Operating Mode")
            gate_df = pd.DataFrame(gate_controller.get_all_gates())
            st.dataframe(gate_df, hide_index=True)


# ==============================================================================
# VIEW 3: Optical Spatial Density & Crush Graph (Replaces Thermal Detection)
# ==============================================================================
elif menu == "🗺️ Optical Spatial Density & Crush Graph":
    st.title("🗺️ Optical CCTV Spatial Density & Proximity Crush Analysis")
    st.info(
        "📷 **No Thermal Hardware Required:** Every analysis below is computed purely from **standard RGB optical CCTV cameras** "
        "using Social Proximity Clustering, Spatial Grid Choke-Point Detection, and Optical Density Gradients."
    )

    datasets = get_available_datasets()
    source_choice = st.radio(
        "Choose Input CCTV Source:",
        ["Sample from Uploaded Crowd Dataset", "Upload Custom Optical Image"],
        horizontal=True,
    )

    image_rgb = None

    if source_choice == "Sample from Uploaded Crowd Dataset" and datasets:
        hc1, hc2 = st.columns(2)
        with hc1:
            sel_dataset = st.selectbox("Dataset:", list(datasets.keys()))
        with hc2:
            meta = get_dataset_metadata(datasets[sel_dataset])
            s_idx = st.number_input(f"Image Index (0 to {len(meta)-1}):", 0, len(meta)-1, 0)

        image_rgb, s_meta = load_dataset_sample(datasets[sel_dataset], int(s_idx))
        st.caption(f"**Image:** `{s_meta['file_name']}` | **Ground Truth Count:** `{s_meta['ground_truth_count']} people`")

    elif source_choice == "Upload Custom Optical Image":
        heat_upload = st.file_uploader("Upload Standard CCTV Image", type=["jpg", "jpeg", "png"])
        if heat_upload:
            pil_img = Image.open(heat_upload).convert("RGB")
            image_rgb = np.array(pil_img)

    if image_rgb is not None:
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            mode_label = st.selectbox("Headcount Accuracy Mode:", list(DETECTION_MODES.keys()), index=0)
            selected_mode = DETECTION_MODES[mode_label]
        with p_c2:
            det_conf = st.slider("Detection Confidence Cutoff:", 0.05, 0.80, 0.15, 0.05)
        with p_c3:
            prox_dist = st.slider("Crush Proximity Threshold (pixels):", 30, 120, 65, 5)

        if st.button("🚀 Run Optical Spatial & Crush Analysis", type="primary"):
            with st.spinner("Running Enhanced Optical Detection & Proximity Clustering..."):
                det_result = detector.detect_people(
                    image_rgb,
                    confidence_threshold=det_conf,
                    mode=selected_mode,
                    draw_annotations=True,
                )
                boxes = det_result["boxes"]

                # 1. Social Proximity & Crush Graph (Optical CCTV Stampede Prevention)
                prox_img, prox_stats = spatial_analyzer.generate_proximity_crush_graph(
                    image_rgb, boxes, distance_threshold_px=prox_dist
                )

                # 2. 4x4 Spatial Choke-Point Grid
                grid_analysis = spatial_analyzer.analyze_spatial_grid(image_rgb, boxes)

                # 3. Optical Density Concentration Overlay (Software-computed)
                opt_density_img, _ = spatial_analyzer.generate_optical_density_map(
                    image_rgb, boxes, gaussian_sigma=25, alpha=0.45
                )

            # Summary Metrics
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Headcount Estimate", f"{det_result['count']} persons", f"{det_result['detected_boxes_count']} boxes")
            sm2.metric("Crush Risk Index", f"{prox_stats['crush_index']}", prox_stats["risk_status"][:20])
            sm3.metric("Choke-Point Sectors", f"{len(grid_analysis['hotspots'])} sectors")
            sm4.metric("Inference Latency", f"{det_result['inference_time_ms']} ms")

            st.divider()
            tab_prox, tab_grid, tab_grad = st.tabs([
                "🔗 Social Proximity & Crush Graph",
                "📐 4x4 Choke-Point Sector Grid",
                "🌈 Software Optical Density Overlay"
            ])

            with tab_prox:
                st.subheader("Social Proximity & Physical Crush Graph")
                st.caption("Connects individuals standing within critical proximity (Red = Extreme crush risk, Yellow = Caution, Green = Safe distance).")
                st.image(prox_img, caption=f"Status: {prox_stats['risk_status']} | Close Contact Pairs: {prox_stats['total_connections']}", use_container_width=True)

            with tab_grid:
                st.subheader("4x4 Spatial Sector Choke-Point Grid")
                st.caption("Divides standard CCTV camera view into 16 physical sectors to pinpoint bottlenecks at gates and turnstiles.")
                st.image(grid_analysis["grid_annotated_image"], caption="Red Highlights = Congested Bottleneck Choke Points", use_container_width=True)

                if grid_analysis["hotspots"]:
                    st.error("⚠️ **Identified Bottleneck Sectors:**")
                    hotspot_df = pd.DataFrame(grid_analysis["hotspots"])
                    st.dataframe(hotspot_df[["cell_name", "count", "crowd_share_pct", "status"]], hide_index=True)

            with tab_grad:
                st.subheader("Software Optical Density Gradient (RGB Camera)")
                st.caption("Calculated purely via software Gaussian density estimation on detected coordinates—no thermal camera needed.")
                st.image(opt_density_img, caption="Optical Density Distribution", use_container_width=True)


# ==============================================================================
# VIEW 4: Crowd Dataset Benchmark (With Enhanced Accuracy Modes)
# ==============================================================================
elif menu == "🖼️ Crowd Dataset Benchmark":
    st.title("🖼️ Uploaded Crowd Dataset Benchmark (Enhanced Headcount)")
    st.markdown(
        "Test our upgraded **High-Density Sliced Inference** and **Optical Calibration** on the **1,198 images** in `Dataset_CrowdPulse/`."
    )

    datasets = get_available_datasets()
    if not datasets:
        st.error("No .parquet datasets found in `Dataset_CrowdPulse/`.")
    else:
        with st.expander("📊 View Complete Dataset Breakdown (1,198 Images)"):
            summary_df = get_dataset_summary()
            st.dataframe(summary_df, hide_index=True)

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_name = st.selectbox("Select Dataset Split:", list(datasets.keys()))
            selected_path = datasets[selected_name]

        metadata_df = get_dataset_metadata(selected_path)
        total_samples = len(metadata_df)

        with col_sel2:
            sample_idx = st.number_input(
                f"Select Image Index (0 to {total_samples - 1}):",
                min_value=0,
                max_value=total_samples - 1,
                value=0,
                step=1,
            )

        meta_row = metadata_df.iloc[sample_idx]
        gt_count = int(meta_row["count"])

        st.caption(
            f"**File:** `{meta_row['file_name']}` | **Split:** `{meta_row['split']}` | "
            f"**Part:** `{meta_row['part']}` | **Ground Truth Count:** `{gt_count} persons`"
        )

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            b_mode_label = st.selectbox("Detection Accuracy Engine:", list(DETECTION_MODES.keys()), index=0)
            b_mode = DETECTION_MODES[b_mode_label]
        with b_c2:
            conf_slider = st.slider("Confidence Threshold (Lower catches small heads):", 0.05, 0.80, 0.15, 0.05)

        if st.button("🚀 Run Enhanced Headcount Detection", type="primary"):
            with st.spinner("Running High-Density Optical Inference..."):
                img_rgb, sample_meta = load_dataset_sample(selected_path, int(sample_idx))
                result = detector.detect_people(
                    image_rgb=img_rgb,
                    confidence_threshold=conf_slider,
                    mode=b_mode,
                    draw_annotations=True,
                )

            pred_count = result["count"]
            abs_diff = abs(pred_count - gt_count)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ground Truth Count", f"{gt_count} persons")
            m2.metric("CrowdPulse Headcount", f"{pred_count} persons", delta=f"{pred_count - gt_count:+d} diff")
            m3.metric("Direct Boxes Found", f"{result['detected_boxes_count']} boxes")
            m4.metric("Inference Time", f"{result['inference_time_ms']} ms", f"on {result['device_used']}")

            st.divider()
            c_orig, c_annot = st.columns(2)
            with c_orig:
                st.subheader("Original Dataset Image")
                st.image(img_rgb, caption=f"Ground Truth: {gt_count} people", use_container_width=True)
            with c_annot:
                st.subheader(f"Detections ({b_mode_label})")
                st.image(result["annotated_image"], caption=f"Estimated: {pred_count} people ({result['risk_badge']})", use_container_width=True)


# ==============================================================================
# VIEW 5: Custom Image Upload
# ==============================================================================
elif menu == "📤 Upload Custom Image":
    st.title("📤 Custom Optical CCTV Image Detection")
    st.markdown("Upload any photo to test enhanced headcount accuracy, proximity crush risk, and multi-agent alerts.")

    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"])

    u_col1, u_col2, u_col3 = st.columns(3)
    with u_col1:
        u_mode_label = st.selectbox("Detection Mode:", list(DETECTION_MODES.keys()), index=0)
        u_mode = DETECTION_MODES[u_mode_label]
    with u_col2:
        u_conf = st.slider("Confidence Cutoff:", 0.05, 0.80, 0.15, 0.05)
    with u_col3:
        u_zone = st.selectbox(
            "Assign Reading to Monitored Zone:",
            [f"{z['name']} (ID: {z['id']}, Cap: {z['capacity_limit']})" for z in config.DEFAULT_ZONES]
        )

    selected_zone_id = config.DEFAULT_ZONES[0]["id"]
    selected_cap = config.DEFAULT_ZONES[0]["capacity_limit"]
    for z in config.DEFAULT_ZONES:
        if z["name"] in u_zone:
            selected_zone_id = z["id"]
            selected_cap = z["capacity_limit"]

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)

        with st.spinner("Running Enhanced Optical Detection & Multi-Agent Evaluation..."):
            result = detector.detect_people(
                image_rgb=image_rgb,
                confidence_threshold=u_conf,
                capacity_limit=selected_cap,
                mode=u_mode,
                draw_annotations=True,
            )

            grid_res = spatial_analyzer.analyze_spatial_grid(image_rgb, result["boxes"])

            agent_output = coordinator.process_zone_update(
                zone_id=selected_zone_id,
                current_count=result["count"],
                capacity_limit=selected_cap,
                bottleneck_detected=grid_res["bottleneck_detected"],
                hotspot_count=len(grid_res["hotspots"]),
                inference_latency_ms=result["inference_time_ms"],
            )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Headcount", f"{result['count']} persons")
        mc2.metric("Occupancy", f"{result['occupancy_pct']}%", result["risk_badge"])
        mc3.metric("Bottleneck Choke", "🚨 Yes" if grid_res["bottleneck_detected"] else "No")
        mc4.metric("Agent Status", "Evaluated ✅")

        st.image(result["annotated_image"], caption="Annotated Optical Crowd Detection Output", use_container_width=True)

        if agent_output["anomaly"]:
            st.error(f"🚨 **{agent_output['anomaly']['severity']} Alert Detected:** {agent_output['anomaly']['description']}")
            if agent_output["actions"]:
                with st.expander("View Agent Action Orders"):
                    for act in agent_output["actions"]:
                        st.write(f"- **{act['target_team']}**: {act['instructions']}")


# ==============================================================================
# VIEW 6: CCTV Video Stream
# ==============================================================================
elif menu == "🎥 CCTV Video Stream":
    st.title("🎥 Real-Time Optical CCTV Video Stream")
    st.markdown("Run automated crowd detection frame-by-frame on standard optical CCTV footage.")

    v_col1, v_col2 = st.columns([2, 1])
    with v_col1:
        video_source_type = st.radio(
            "Video Source:",
            ["Simulated Concourse CCTV (Built-in Demo)", "Upload Your Own Video File (.mp4, .avi)"],
            horizontal=True,
        )

    uploaded_video = None
    if video_source_type == "Upload Your Own Video File (.mp4, .avi)":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    with v_col2:
        frame_skip = st.slider("Frame Skip (Higher = faster playback):", 1, 5, 2)
        v_conf = st.slider("Video Confidence:", 0.10, 0.80, 0.18, 0.05)

    if st.button("▶️ Start Video Detection Stream", type="primary"):
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = Path(tfile.name)
        else:
            with st.spinner("Generating simulated CCTV demo feed..."):
                video_path = generate_synthetic_demo_video()

        processor = VideoStreamProcessor(detector=detector)

        video_placeholder = st.empty()
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        m_count = stat_col1.empty()
        m_avg = stat_col2.empty()
        m_peak = stat_col3.empty()
        m_fps = stat_col4.empty()

        progress_text = st.empty()

        for frame_annotated, stats in processor.process_video_file(
            video_path=video_path,
            confidence_threshold=v_conf,
            capacity_limit=25,
            frame_skip=frame_skip,
            max_frames=120,
        ):
            video_placeholder.image(frame_annotated, channels="RGB", use_container_width=True)
            m_count.metric("Frame Count", f"{stats['current_count']} persons", stats["risk_badge"])
            m_avg.metric("Average Count", f"{stats['average_count']} persons")
            m_peak.metric("Peak Count", f"{stats['peak_count']} persons")
            m_fps.metric("Processing Speed", f"{stats['current_fps']} FPS", f"{stats['inference_time_ms']} ms")
            progress_text.caption(f"Processed Frame: #{stats['processed_index']}")

        st.success("✅ Video stream playback finished!")


# ==============================================================================
# VIEW 7: Incident Logs & Historical Analytics
# ==============================================================================
elif "Incident Logs & Analytics" in menu:
    st.title("📈 Incident Logs & Temporal Crowd Analytics")
    st.markdown("Query the **SQLite database** (`crowdpulse.db`) for crowd patterns, audit trails, and alerts.")

    tab_time, tab_alerts, tab_table = st.tabs(["📊 Time-Series Trends", "🚨 Active Safety Incidents", "📋 Raw Database Logs"])

    with tab_time:
        st.subheader("Temporal Crowd Density Trends")
        filter_zone = st.selectbox("Filter by Monitored Zone:", ["All Zones"] + [z["name"] for z in config.DEFAULT_ZONES])

        selected_zid = None
        if filter_zone != "All Zones":
            for z in config.DEFAULT_ZONES:
                if z["name"] == filter_zone:
                    selected_zid = z["id"]

        logs_df = get_recent_logs_dataframe(zone_id=selected_zid, limit=200)

        if logs_df.empty:
            st.info("No historical logs recorded yet.")
        else:
            fig = px.line(
                logs_df,
                x="Timestamp",
                y="Count",
                color="Zone Name",
                markers=True,
                title="Crowd Count Timeline across Monitored Zones",
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            fig_occ = px.area(
                logs_df,
                x="Timestamp",
                y="Occupancy (%)",
                color="Zone Name",
                title="Occupancy Percentage vs Time",
            )
            fig_occ.add_hline(y=config.DENSITY_THRESHOLDS["MODERATE"], line_dash="dash", line_color="orange", annotation_text="Caution (80%)")
            fig_occ.add_hline(y=100.0, line_dash="dash", line_color="red", annotation_text="100% Safe Capacity")
            st.plotly_chart(fig_occ, use_container_width=True)

    with tab_alerts:
        st.subheader("Active & Pending Incident Alerts")
        current_alerts = get_active_alerts()

        if not current_alerts:
            st.success("🎉 No active safety incidents! All zones are operating within safe parameters.")
        else:
            for alert in current_alerts:
                sev_color = "red" if alert["severity"] in ["CRITICAL", "HIGH"] else "orange"
                with st.container(border=True):
                    al_col1, al_col2 = st.columns([3, 1])
                    with al_col1:
                        st.markdown(f"#### 🚨 :{sev_color}[{alert['severity']} ALERT] — {alert['zone_name']}")
                        st.caption(f"**Type:** `{alert['alert_type']}` | **Time:** `{alert['timestamp']}` | **Status:** `{alert['status']}`")
                        st.write(alert["description"])
                        st.write(f"**People Count:** `{alert['person_count']} / {alert['capacity_limit']} persons`")
                    with al_col2:
                        st.write("")
                        if alert["status"] == "ACTIVE":
                            if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                                update_alert_status(alert["id"], "ACKNOWLEDGED")
                                st.rerun()
                        if st.button("Resolve Incident", key=f"res_{alert['id']}", type="primary"):
                            update_alert_status(alert["id"], "RESOLVED")
                            st.rerun()

    with tab_table:
        st.subheader("Recent Database Density Records")
        all_logs = get_recent_logs_dataframe(limit=100)
        if not all_logs.empty:
            st.dataframe(all_logs, use_container_width=True, hide_index=True)
            csv_data = all_logs.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Logs as CSV", csv_data, "crowdpulse_logs.csv", "text/csv")


# ==============================================================================
# VIEW 8: Hardware & Diagnostics
# ==============================================================================
elif menu == "🖥️ Hardware & Environment":
    st.title("🖥️ Hardware & Environment Diagnostics")
    report = get_environment_report()
    sys_info = report["system"]
    pyt_info = report["pytorch"]
    smi = report["nvidia_smi"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Platform", sys_info["os_name"], sys_info["os_release"])
    c2.metric("Python", sys_info["python_version"], "Supported")
    c3.metric("PyTorch Engine", "CUDA (GPU)" if pyt_info["cuda_available"] else "CPU Mode")
    c4.metric("NVIDIA Hardware", "Detected" if smi["available"] else "Not Found")

    st.divider()
    st.markdown("### PyTorch & GPU Acceleration Guide")
    if pyt_info["cuda_available"]:
        st.success("🚀 PyTorch CUDA is operational! All YOLO detections will utilize the NVIDIA GPU.")
    else:
        st.warning(
            "⚠️ PyTorch is currently running on **CPU**.\n\n"
            "To enable 5x-10x faster inference and model fine-tuning with your NVIDIA GeForce RTX 3050 Laptop GPU, "
            "run the official PyTorch CUDA install command in PowerShell:"
        )
        st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124", language="powershell")

    with st.expander("View Raw NVIDIA-SMI Output"):
        st.code(smi.get("raw", "nvidia-smi not available"), language="bash")
