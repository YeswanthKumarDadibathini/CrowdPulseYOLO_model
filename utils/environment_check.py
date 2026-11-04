"""
CrowdPulse - Environment Inspection Utility
Project: CrowdPulse - Crowd Density Management System (SIH25165)
Phase: Phase 1 Foundation

This module inspects the local runtime environment:
1. Python version & OS information
2. Core dependencies (Streamlit, Ultralytics, OpenCV, NumPy, Pandas, etc.)
3. PyTorch installation & CUDA / NVIDIA GPU capability

Beginner Note:
- PyTorch can be built with CPU support or CUDA (GPU) support.
- Having an NVIDIA GPU does NOT automatically mean PyTorch can use it.
- PyTorch must be installed with the matching CUDA version wheels.
"""

import os
import platform
import subprocess
import sys
from importlib.metadata import distributions, version


def check_nvidia_smi() -> dict:
    """Checks if the NVIDIA System Management Interface (nvidia-smi) is present

    and extracts basic GPU driver information.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return {
                "available": True,
                "output_snippet": "\n".join(lines[:12]),
                "raw": result.stdout,
            }
        return {"available": False, "error": f"nvidia-smi exited with code {result.returncode}"}
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi tool not found in system PATH"}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def get_environment_report() -> dict:
    """Collects and returns comprehensive hardware and library status as a dictionary."""
    report = {}

    # 1. System & Python
    report["system"] = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "python_executable": sys.executable,
    }

    # 2. Key Phase 1 Libraries
    packages_to_check = [
        "streamlit",
        "ultralytics",
        "opencv-python",
        "numpy",
        "pandas",
        "plotly",
        "sqlalchemy",
        "pillow",
        "scipy",
        "pytest",
        "torch",
        "torchvision",
    ]

    installed_packages = {}
    for pkg in packages_to_check:
        try:
            installed_packages[pkg] = {
                "installed": True,
                "version": version(pkg),
            }
        except Exception:
            installed_packages[pkg] = {
                "installed": False,
                "version": None,
            }

    report["packages"] = installed_packages

    # 3. PyTorch and CUDA Inspection
    pytorch_info = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version_built": None,
        "device_count": 0,
        "devices": [],
        "note": "",
    }

    try:
        import torch

        pytorch_info["installed"] = True
        pytorch_info["version"] = torch.__version__
        pytorch_info["cuda_available"] = torch.cuda.is_available()
        pytorch_info["cuda_version_built"] = torch.version.cuda

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            pytorch_info["device_count"] = count
            for i in range(count):
                device_name = torch.cuda.get_device_name(i)
                total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                pytorch_info["devices"].append({
                    "index": i,
                    "name": device_name,
                    "total_memory_gb": round(total_mem_gb, 2),
                })
            pytorch_info["note"] = f"PyTorch is successfully utilizing CUDA with {count} GPU(s)."
        else:
            pytorch_info["note"] = (
                "PyTorch is running in CPU mode. (torch.cuda.is_available() is False)"
            )
    except ImportError:
        pytorch_info["note"] = "PyTorch is not installed in the current environment."
    except Exception as exc:
        pytorch_info["note"] = f"Error inspecting PyTorch: {exc}"

    report["pytorch"] = pytorch_info

    # 4. System-level NVIDIA SMI
    report["nvidia_smi"] = check_nvidia_smi()

    return report


def print_environment_summary():
    """Prints a friendly terminal report for students and reviewers."""
    data = get_environment_report()
    sys_info = data["system"]
    pyt = data["pytorch"]
    smi = data["nvidia_smi"]

    print("=" * 65)
    print("      CrowdPulse - System & Environment Inspection (Phase 1)")
    print("=" * 65)
    print(f"OS Platform       : {sys_info['platform']}")
    print(f"Python Version    : {sys_info['python_version']}")
    print(f"Python Executable : {sys_info['python_executable']}")
    print("-" * 65)

    print("Core Package Versions:")
    for pkg_name, details in data["packages"].items():
        if details["installed"]:
            print(f"  - {pkg_name:<16}: {details['version']}")
        else:
            print(f"  - {pkg_name:<16}: NOT INSTALLED")

    print("-" * 65)
    print("NVIDIA Hardware & PyTorch CUDA Status:")
    if smi["available"]:
        print("  - NVIDIA Driver detected via nvidia-smi: YES")
    else:
        print(f"  - NVIDIA Hardware status: {smi.get('error', 'Unknown')}")

    print(f"  - PyTorch Version    : {pyt.get('version', 'None')}")
    print(f"  - Built with CUDA   : {pyt.get('cuda_version_built', 'None (CPU only)')}")
    print(f"  - CUDA Available     : {'YES [GPU Ready]' if pyt['cuda_available'] else 'NO [CPU Mode]'}")

    if pyt["cuda_available"]:
        for dev in pyt["devices"]:
            print(f"    * GPU #{dev['index']}: {dev['name']} ({dev['total_memory_gb']} GB VRAM)")
    else:
        if smi["available"]:
            print("\n  [!] NOTICE: NVIDIA GPU was detected on the laptop, but PyTorch")
            print("      is currently installed without CUDA acceleration support.")
            print("      To enable NVIDIA GPU acceleration, run:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    print("=" * 65)


if __name__ == "__main__":
    print_environment_summary()

