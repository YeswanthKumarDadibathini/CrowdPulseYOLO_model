"""
CrowdPulse - Crowd Dataset Loader (Phase 2)
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 2 Vision & Detection Pipeline

Beginner-Friendly Explanation:
This module connects to the 4 Parquet files in the 'Dataset_CrowdPulse' folder.
Inside each Parquet file, images are stored as raw bytes alongside their ground-truth
people count, dataset part (Part A dense / Part B sparse), and split (train/test).

We use:
- pandas: to read the tabular metadata from .parquet files
- io.BytesIO + PIL (Pillow): to convert raw image bytes into image arrays
- numpy: to convert images into RGB arrays for OpenCV and Streamlit
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image

import config

# Directory containing the uploaded dataset files
DATASET_DIR = config.BASE_DIR / "Dataset_CrowdPulse"


def get_available_datasets() -> Dict[str, Path]:
    """Finds all available .parquet dataset files in Dataset_CrowdPulse/.

    Returns a dictionary mapping a friendly name to its Path.
    """
    if not DATASET_DIR.exists():
        return {}

    parquet_files = sorted(DATASET_DIR.glob("*.parquet"))
    datasets = {}
    for p in parquet_files:
        # Create a readable name from filename, e.g. "Part A (Test) - 182 imgs"
        stem = p.stem.lower()
        if "part_a_2" in stem:
            name = "Part A - Train (Dense Crowds, 300 imgs)"
        elif "part_a" in stem:
            name = "Part A - Test (Dense Crowds, 182 imgs)"
        elif "part_b_2" in stem:
            name = "Part B - Train (Sparse Crowds, 400 imgs)"
        elif "part_b" in stem:
            name = "Part B - Test (Sparse Crowds, 316 imgs)"
        else:
            name = p.name
        datasets[name] = p

    return datasets


def get_dataset_metadata(file_path: Path) -> pd.DataFrame:
    """Reads non-image metadata columns from a parquet file quickly.

    We exclude the heavy 'image' column so it loads instantly.
    """
    return pd.read_parquet(
        file_path,
        columns=["file_name", "count", "part", "split"]
    )


def load_dataset_sample(
    file_path: Path,
    index: int = 0
) -> Tuple[np.ndarray, Dict]:
    """Loads a single image and its ground-truth metadata by row index.

    Parameters:
        file_path (Path): Path to the parquet file
        index (int): Row number (0 to len - 1)

    Returns:
        tuple: (image_rgb_numpy_array, metadata_dictionary)
    """
    # Read only the selected row to conserve RAM
    df = pd.read_parquet(file_path)
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of bounds for dataset of length {len(df)}")

    row = df.iloc[index]

    # The image is stored as a dict: {'bytes': b'...', 'path': '...'}
    image_dict = row["image"]
    image_bytes = image_dict["bytes"]

    # Decode bytes into a PIL Image and convert to RGB NumPy array
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)

    metadata = {
        "file_name": row["file_name"],
        "ground_truth_count": int(row["count"]),
        "part": str(row["part"]),
        "split": str(row["split"]),
        "image_width": img_rgb.shape[1],
        "image_height": img_rgb.shape[0],
    }

    return img_rgb, metadata


def get_dataset_summary() -> pd.DataFrame:
    """Generates a summary table of all parquet datasets."""
    datasets = get_available_datasets()
    records = []

    for name, path in datasets.items():
        meta = get_dataset_metadata(path)
        records.append({
            "Dataset Split": name,
            "Total Images": len(meta),
            "Min Count": int(meta["count"].min()),
            "Max Count": int(meta["count"].max()),
            "Average Count": round(float(meta["count"].mean()), 1),
            "File Name": path.name,
        })

    return pd.DataFrame(records)

