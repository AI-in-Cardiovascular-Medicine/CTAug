import pytest
import torch

monai = pytest.importorskip("monai")

from ctaug.monai import (
    Calcificationd,
    Metald,
    Motiond,
    RandomArtifactd,
    Stepd,
    StepMotiond,
    Wired,
)

SPACING = (1.5, 1.0, 1.0)


@pytest.fixture
def data():
    torch.manual_seed(0)
    return {
        "image": torch.rand(1, 24, 32, 32) * 500,
        "label": torch.randint(0, 4, (1, 24, 32, 32)),
    }


@pytest.mark.parametrize("cls", [Metald, Wired, Calcificationd])
def test_artifact_adapters(data, cls):
    transform = cls(keys=["image"], spacing=SPACING, label_key="label", max_n_specs=1, prob=1.0)
    out = transform(dict(data))
    assert out["image"].shape == data["image"].shape


def test_random_artifact_adapter(data):
    transform = RandomArtifactd(keys=["image"], spacing=SPACING, label_key="label", max_n_specs=1, prob=1.0)
    out = transform(dict(data))
    assert out["image"].shape == data["image"].shape


def test_step_and_shoot_adapter(data):
    out = Stepd(keys=["image"], prob=1.0)(dict(data))
    assert out["image"].shape == data["image"].shape


def test_motion_adapter(data):
    out = Motiond(keys=["image"], label_key="label", prob=1.0)(dict(data))
    assert out["image"].shape == data["image"].shape
    assert out["label"].shape == data["label"].shape


def test_motion_adapter_requires_label(data):
    with pytest.raises(KeyError):
        Motiond(keys=["image"], label_key="missing", prob=1.0)({"image": data["image"]})


def test_prob_zero_is_passthrough(data):
    out = Stepd(keys=["image"], prob=0.0)(dict(data))
    assert torch.equal(out["image"], data["image"])
