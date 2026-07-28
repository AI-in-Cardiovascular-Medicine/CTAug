import random

import numpy as np
import pytest
import torch

from ctaug import (
    CalcificationTransform,
    MetalTransform,
    MotionTransform,
    RandomArtifactTransform,
    StepMotionTransform,
    StepTransform,
    WireTransform,
)

from ctaug.step.functional import motion_core

SPACING = (1.5, 1.0, 1.0)


def _axis_slice(axis, sl, ndim=3):
    """Index tuple selecting ``sl`` along ``axis`` and everything else."""
    idx = [slice(None)] * ndim
    idx[axis] = sl
    return tuple(idx)


@pytest.fixture
def volume():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    image = torch.rand(1, 24, 32, 32) * 500
    segmentation = torch.randint(0, 4, (1, 24, 32, 32))
    return image, segmentation


@pytest.mark.parametrize("cutoff_index, move_index",
                         [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
@pytest.mark.parametrize("move_left", [True, False])
def test_motion_core_translates_the_cut_block(cutoff_index, move_index, move_left):
    """Every axis combination must translate the block past the cut, not duplicate or erase it."""
    shape = (6, 7, 8)
    vol = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    cut, move = 2, 3

    out = motion_core(vol, cutoff_index=cutoff_index, move_index=move_index, cropped_value="zero",
                      cut_off_position=cut, img_size=shape, mean_value=0.0, move_value=move,
                      move_left=move_left, input_type="image")
    assert out.shape == vol.shape

    # the block at/after the cut is shifted along move_index, and the vacated strip zero-filled
    block = _axis_slice(cutoff_index, slice(cut, None))
    if move_left:
        src, dst, pad = slice(move, None), slice(None, -move), slice(-move, None)
    else:
        src, dst, pad = slice(None, -move), slice(move, None), slice(None, move)
    moved, original = out[block], vol[block]
    np.testing.assert_array_equal(moved[_axis_slice(move_index, dst)],
                                  original[_axis_slice(move_index, src)])
    np.testing.assert_array_equal(moved[_axis_slice(move_index, pad)], 0)

    # everything before the cut is left alone
    kept = _axis_slice(cutoff_index, slice(None, cut))
    np.testing.assert_array_equal(out[kept], vol[kept])


@pytest.mark.parametrize("transform_cls", [MetalTransform, WireTransform, CalcificationTransform])
@pytest.mark.parametrize("label_kwargs", [
    {"exclude_labels": 0},                              # int instead of a sequence
    {"include_labels": 2},                              # int instead of a sequence
    {"exclude_labels": None, "include_labels": None},   # no filter at all
    {"exclude_labels": (0,), "include_labels": None},   # the default
])
def test_artifact_transform_label_filters(volume, transform_cls, label_kwargs):
    image, segmentation = volume
    out = transform_cls(spacing=SPACING, max_n_specs=1, **label_kwargs)(
        image=image, segmentation=segmentation)
    assert out["image"].shape == image.shape


@pytest.mark.parametrize("transform_cls", [MetalTransform, WireTransform, CalcificationTransform])
def test_artifact_transform_preserves_shape(volume, transform_cls):
    image, segmentation = volume
    transform = transform_cls(spacing=SPACING, max_n_specs=1)
    out = transform(image=image, segmentation=segmentation)
    assert out["image"].shape == image.shape
    assert f"{transform_cls.__name__}_info" in out


def test_random_artifact_transform(volume):
    image, segmentation = volume
    transform = RandomArtifactTransform(spacing=SPACING, max_n_specs=1)
    out = transform(image=image, segmentation=segmentation)
    assert out["image"].shape == image.shape


def test_step_and_shoot_transform(volume):
    image, _ = volume
    out = StepTransform()(image=image)
    assert out["image"].shape == image.shape


def test_motion_transform(volume):
    image, segmentation = volume
    out = MotionTransform()(image=image, segmentation=segmentation)
    assert out["image"].shape == image.shape
    assert out["segmentation"].shape == segmentation.shape


def test_step_motion_transform(volume):
    image, segmentation = volume
    out = StepMotionTransform()(image=image, segmentation=segmentation)
    assert out["image"].shape == image.shape
    assert out["segmentation"].shape == segmentation.shape
    assert out["segmentation"].dtype == segmentation.dtype
    assert "StepMotionTransform_info" in out


def test_step_motion_transform_without_segmentation(volume):
    image, _ = volume
    out = StepMotionTransform()(image=image)
    assert out["image"].shape == image.shape
    assert "segmentation" not in out


def test_step_motion_transform_shifts_image(volume):
    """The transform must actually change the volume, not pass it through."""
    image, segmentation = volume
    out = StepMotionTransform()(image=image, segmentation=segmentation)
    assert not torch.equal(out["image"], image)


@pytest.mark.parametrize("transform_cls", [StepTransform, MotionTransform, StepMotionTransform])
@pytest.mark.parametrize("channel_first", [True, False])
def test_step_transforms_round_trip_numpy(volume, transform_cls, channel_first):
    """Plain numpy in/out, with and without a leading channel axis."""
    image, segmentation = volume
    img = image.numpy() if channel_first else image.numpy()[0]
    seg = segmentation.numpy() if channel_first else segmentation.numpy()[0]
    out = transform_cls()(image=img, segmentation=seg)
    assert isinstance(out["image"], np.ndarray)
    assert out["image"].shape == img.shape
    assert out["segmentation"].shape == seg.shape
    assert out["segmentation"].dtype == seg.dtype
