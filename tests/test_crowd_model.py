"""
CrowdPulse - Deep Crowd Counter & Training Pipeline Unit Tests
Project: CrowdPulse - Crowd Density Management System (SIH25165)

Run using:
pytest tests/test_crowd_model.py -v
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from detection.crowd_counter_model import CrowdCountPredictor, CrowdPulseCountNet
from training.train_crowd_model import ParquetCrowdDataset


class TestCrowdPulseCountNet:
    """Verifies pretrained transfer learning architecture, gradient control, and count bounds."""

    @pytest.mark.parametrize("backbone", ["convnext_tiny", "resnet50", "efficientnet_b0"])
    def test_backbone_forward_and_non_negative_count(self, backbone):
        model = CrowdPulseCountNet(backbone_name=backbone, pretrained=False)
        model.eval()

        dummy_batch = torch.randn(2, 3, 256, 256)
        log_out = model(dummy_batch)
        assert log_out.shape == (2,)

        counts = model.predict_count(dummy_batch)
        assert counts.shape == (2,)
        assert torch.all(counts >= 0.0)

    def test_freeze_and_unfreeze_backbone(self):
        model = CrowdPulseCountNet(backbone_name="efficientnet_b0", pretrained=False)

        model.freeze_backbone()
        assert all(not p.requires_grad for p in model.features.parameters())
        assert all(p.requires_grad for p in model.head.parameters())

        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.features.parameters())

    def test_invalid_backbone_raises(self):
        with pytest.raises(ValueError):
            CrowdPulseCountNet(backbone_name="unknown_net", pretrained=False)

    def test_checkpoint_round_trip_and_tta_inference(self, tmp_path):
        ckpt_file = tmp_path / "test_model.pth"
        model = CrowdPulseCountNet(backbone_name="efficientnet_b0", pretrained=False)

        torch.save(
            {
                "backbone_name": "efficientnet_b0",
                "img_size": 224,
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "metrics": {"overall_mae": 12.5},
            },
            ckpt_file,
        )

        predictor = CrowdCountPredictor(checkpoint_path=ckpt_file)
        assert predictor.is_ready
        assert predictor.backbone_name == "efficientnet_b0"

        dummy_rgb = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        res = predictor.predict(dummy_rgb, use_tta=True)
        assert isinstance(res["predicted_count"], int)
        assert res["predicted_count"] >= 0
        assert res["tta_used"] is True


class TestDatasetLeakageGuard:
    """Verifies strict train/test isolation prevents data leakage."""

    def test_rejects_test_file_in_train_dataset(self):
        test_parquet = PROJECT_ROOT / "Dataset_CrowdPulse" / "project_CrowdPulse_part_A.parquet"
        if not test_parquet.exists():
            pytest.skip("Dataset_CrowdPulse not available locally")

        with pytest.raises(ValueError, match="Data leakage guard"):
            ParquetCrowdDataset([test_parquet], expected_split="train")

