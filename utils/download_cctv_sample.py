"""
Download sample real-world CCTV surveillance footage for testing CrowdPulse.
"""
import os
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "name": "real_cctv_vtest.avi",
        "url": "https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi",
        "description": "Official OpenCV CCTV Surveillance benchmark video (concourse pedestrian traffic)"
    },
    {
        "name": "real_cctv_pedestrians.mp4",
        "url": "https://github.com/nandinib1999/object-detection-yolo-opencv/raw/master/videos/pedestrians.mp4",
        "description": "Pedestrian street crosswalk & concourse CCTV footage"
    }
]

def download_samples():
    for item in SAMPLES:
        dest = DATA_DIR / item["name"]
        print(f"[INFO] Checking {item['name']}...")
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"[OK] {item['name']} already exists ({dest.stat().st_size} bytes).")
            continue
        print(f"[DOWNLOADING] {item['url']} -> {dest} ...")
        try:
            req = urllib.request.Request(
                item["url"],
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest, "wb") as out_file:
                data = response.read()
                out_file.write(data)
            print(f"[SUCCESS] Downloaded {item['name']} ({dest.stat().st_size} bytes).")
        except Exception as e:
            print(f"[ERROR] Failed to download {item['name']}: {e}")

if __name__ == "__main__":
    download_samples()
