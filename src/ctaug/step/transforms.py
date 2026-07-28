from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Union

import torch

from ctaug._base import DictTransform
from ctaug.step.functional import (
    MotionAugmentation,
    StepAugmentation,
    StepMotionAugmentation,
)


class StepTransform(DictTransform):
    """Dict-transform wrapper around :class:`ctaug.motion.functional.StepAugmentation`.

    Expects ``data_dict[key_origin]`` to be a torch tensor shaped ``(1, Z, Y, X)``.
    """

    def __init__(self, key_origin: str = "image", key_target: str = "image",
                cutoff_index: int = -1,
                cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                normalize: bool = True):
        self.key_target = key_target
        self.key_origin = key_origin
        self.augmentor = StepAugmentation(cutoff_index=cutoff_index,
                                          cut_off_pixel_value_weight=cut_off_pixel_value_weight,
                                          normalize=normalize)

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict[self.key_origin]
        to_torch = False
        if isinstance(data, torch.Tensor):
            data = data.numpy().copy()
            to_torch = True
        
        do_unsqueeze = False
        if len(data.shape) > 3:
            if data.shape[0] == 1:
                data = data.squeeze()
                do_unsqueeze = True
            if len(data.shape) > 3:
                raise ValueError(f"Input data shape is more than 3 or more than 4 while first one is 1, data shape: {data.shape}")
        initial_size = data.shape
        data, info = self.augmentor(data)
        if data.shape != initial_size:
            raise ValueError(f"Error in StepTransform for size, {data.shape} and {initial_size}")
        if to_torch:
            data = torch.from_numpy(data)
        if do_unsqueeze:
            data = data[None, ...]
        data_dict[self.key_target] = data
        data_dict[f"{self.__class__.__name__}_info"] = info
        
        return data_dict


class MotionTransform(DictTransform):
    """Dict-transform wrapper around :class:`ctaug.motion.functional.MotionAugmentation`.

    Requires both ``data_dict[key_origin]`` and ``data_dict["segmentation"]`` (torch tensors
    shaped ``(1, Z, Y, X)``); the segmentation is shifted alongside the image.
    """

    def __init__(self, key_origin: str = "image", key_target: str = "image",
                motion_move_position: Union[Tuple[float, float], List[float]] = (0.05, 0.2),
                motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.015),
                normalize: bool = True):
        self.key_target = key_target
        self.key_origin = key_origin
        self.augmentor = MotionAugmentation(motion_move_position=motion_move_position,
                                            motion_move_range=motion_move_range,
                                            normalize=normalize)

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict[self.key_origin]
        seg = data_dict["segmentation"]
        seg_dtype = seg.dtype
        to_torch = False
        if isinstance(data, torch.Tensor):
            data = data.numpy().copy()
            to_torch = True
        seg_to_torch = False
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy().copy()
            seg_to_torch = True
        if data.shape != seg.shape:
            raise ValueError(f"Seg and data don't have the same shape, data: {data.shape}, seg: {seg.shape}")

        do_unsqueeze = False
        if len(data.shape) > 3:
            if data.shape[0] == 1:
                data = data.squeeze()
                seg = seg.squeeze()
                do_unsqueeze = True
            if len(data.shape) > 3:
                raise ValueError(f"Input data shape is more than 3 or more than 4 while first one is 1, data shape: {data.shape}")
        initial_size = data.shape
        data, seg, info = self.augmentor(data, seg, random.choice(["zero", "mean", "cut_resize"]))
        if data.shape != initial_size:
            raise ValueError(f"Error in MotionTransform for size, {data.shape} and {initial_size}")
        if to_torch:
            data = torch.from_numpy(data)
        if seg_to_torch:
            seg = torch.from_numpy(seg).to(seg_dtype)
        else:
            seg = seg.astype(seg_dtype)
        if do_unsqueeze:
            data = data[None, ...]
            seg = seg[None, ...]
        data_dict[self.key_target] = data
        data_dict["segmentation"] = seg
        data_dict[f"{self.__class__.__name__}_info"] = info
        return data_dict


class StepMotionTransform(DictTransform):
    """Dict-transform wrapper around :class:`ctaug.motion.functional.StepMotionAugmentation`.

    Expects ``data_dict[key_origin]`` as a torch tensor shaped ``(1, Z, Y, X)``; ``data_dict["segmentation"]``
    is used and shifted alongside the image when present, but is optional.
    """

    def __init__(self, key_origin: str = "image", key_target: str = "image",
                cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.025),
                normalize: bool = True):
        self.key_target = key_target
        self.key_origin = key_origin
        self.augmentor = StepMotionAugmentation(cut_off_pixel_value_weight=cut_off_pixel_value_weight,
                                                motion_move_range=motion_move_range,
                                                normalize=normalize)

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict[self.key_origin]
        seg = data_dict["segmentation"] if "segmentation" in data_dict else None
        seg_dtype = seg.dtype if seg is not None else None
        to_torch = False
        if isinstance(data, torch.Tensor):
            data = data.numpy().copy()
            to_torch = True
        seg_to_torch = False
        if seg is not None and isinstance(seg, torch.Tensor):
            seg = seg.numpy().copy()
            seg_to_torch = True
        if seg is not None and data.shape != seg.shape:
            raise ValueError(f"Seg and data don't have the same shape, data: {data.shape}, seg: {seg.shape}")

        do_unsqueeze = False
        if len(data.shape) > 3:
            if data.shape[0] == 1:
                data = data.squeeze()
                if seg is not None:
                    seg = seg.squeeze()
                do_unsqueeze = True
            if len(data.shape) > 3:
                raise ValueError(f"Input data shape is more than 3 or more than 4 while first one is 1, data shape: {data.shape}")
        initial_size = data.shape

        data, seg, info = self.augmentor(data, seg)
        if data.shape != initial_size:
            raise ValueError(f"Error in StepMotionTransform for size, {data.shape} and {initial_size}")

        if to_torch:
            data = torch.from_numpy(data)
        if do_unsqueeze:
            data = data[None, ...]
        data_dict[self.key_target] = data
        if seg is not None:
            if seg_to_torch:
                seg = torch.from_numpy(seg).to(seg_dtype)
            else:
                seg = seg.astype(seg_dtype)
            if do_unsqueeze:
                seg = seg[None, ...]
            data_dict["segmentation"] = seg
        data_dict[f"{self.__class__.__name__}_info"] = info
        return data_dict
