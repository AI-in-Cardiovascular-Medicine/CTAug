import random
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ctaug._base import DictTransform
from ctaug.metal.functional import (
    mask_base_position_2d,
    mask_base_position_3d,
    simulate_artifacts_unified,
)

ValueOrRange = Union[int, float, Tuple[float, float], Sequence[float]]


def sample_value(value: ValueOrRange) -> Union[int, float]:
    """Resolve a scalar-or-range argument of an artifact transform into a single value.

    A scalar is used as-is; a ``(low, high)`` pair is sampled per call, with integer bounds
    drawing an integer.
    """
    if isinstance(value, (int, float)):
        return value
    low, high = value
    if isinstance(low, int) and isinstance(high, int):
        return random.randint(low, high)
    return random.uniform(low, high)


class MetalTransform(DictTransform):
    """Insert one or more 2D metal implants (circle/ellipse/rectangle) and simulate streak artifacts.

    Expects ``data_dict[key_origin]`` to be a torch tensor shaped ``(1, Z, Y, X)``, and, if
    available, ``data_dict["segmentation"]`` with the same spatial shape, used to anchor
    implant placement on a random foreground label.
    """

    def __init__(
        self,
        spacing,
        key_origin: str = "image",
        key_target: str = "image",
        max_n_specs: int = 2,
        severity: ValueOrRange = (0.1, 0.9),
        intensity: ValueOrRange = (1000, 5000),
        exclude_labels: Union[Sequence, int, None] = (0,),
        include_labels: Union[Sequence, int, None] = None,
        verbose: bool = False,
        skip_minus_one_selected_label: bool = True,
        radius_min: Tuple[int, int] = (0.35, 0.35),
        radius_max: Tuple[int, int] = (3.0, 3.0),
        denormalize_mean: Optional[float] = None,
        denormalize_std: Optional[float] = None,
    ):
        self.key_target = key_target
        self.key_origin = key_origin
        self.spacing = spacing
        self.max_n_specs = max_n_specs
        self.severity = severity
        self.intensity = intensity
        self.exclude_labels = exclude_labels
        self.include_labels = include_labels
        self.verbose = verbose
        self.skip_minus_one_selected_label = skip_minus_one_selected_label
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.denormalize_mean = denormalize_mean
        self.denormalize_std = denormalize_std

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict[self.key_origin]
        data_dtype = data.dtype
        to_torch = False
        if isinstance(data, torch.Tensor):
            data = data.numpy().copy().astype(np.float32)
            to_torch = True
        seg = data_dict.get("segmentation")
        if seg is not None and isinstance(seg, torch.Tensor):
            seg = seg.numpy().copy()
        if seg is not None and data.shape != seg.shape:
            raise ValueError(
                f"Seg and data don't have the same shape, data: {data.shape}, seg: {seg.shape}"
            )
        do_unsqueeze = False
        if len(data.shape) > 3:
            if data.shape[0] == 1:
                data = data.squeeze(0)
                if seg is not None:
                    seg = seg.squeeze(0)
                do_unsqueeze = True
            if len(data.shape) > 3:
                raise ValueError(
                    f"Input data shape is more than 3 or more than 4 while first one is 1, data shape: {data.shape}"
                )
        if self.denormalize_mean is not None and self.denormalize_std is not None:
            data = data * self.denormalize_std + self.denormalize_mean
        severity = sample_value(self.severity)
        data_shape = data.shape
        data_min, data_max = data.min(), data.max()

        n_specs = random.randint(1, max(1, self.max_n_specs))
        implant_specs = []
        selected_labels = []

        for _ in range(n_specs):
            max_slice = round(data_shape[0] * 0.2)
            (start_slice, end_slice), center, selected_label = mask_base_position_2d(
                seg,
                data,
                max_slice,
                exclude_labels=self.exclude_labels,
                include_labels=self.include_labels,
                verbose=self.verbose,
            )

            if selected_label != -1 or not self.skip_minus_one_selected_label:
                rlo_min, clo_min = (
                    max(self.spacing[1], self.radius_min[0]),
                    max(self.spacing[2], self.radius_min[1] or 0),
                )
                rlo_max, clo_max = (
                    max(5 * rlo_min, self.radius_max[0]),
                    max(5 * clo_min, self.radius_max[1]),
                )

                radius = (
                    random.uniform(rlo_min, rlo_max),
                    random.uniform(clo_min, clo_max),
                )
                spec = {
                    "type": "2d",
                    "shape": random.choice(("circle", "ellipse", "rectangle")),
                    "slices": list(range(start_slice, end_slice)),
                    "center_px": (int(center[0]), int(center[1])),
                    "radius_mm": radius,
                    "intensity": sample_value(self.intensity),
                }
                if selected_label != -1:
                    selected_labels.append(selected_label)
                implant_specs.append(spec)

        angle_size = random.randint(180, 720)
        if implant_specs:
            angles = np.linspace(0, 180, angle_size, endpoint=False)
            data = simulate_artifacts_unified(
                data,
                implant_specs,
                angles=angles,
                severity=severity,
                spacing=self.spacing,
                verbose=self.verbose,
            )
            data = np.clip(data, a_min=data_min, a_max=data_max)
        if do_unsqueeze:
            data = data[None, ...]
        if to_torch:
            data = torch.from_numpy(data).to(data_dtype)
        else:
            data = data.astype(data_dtype)

        data_dict[self.key_target] = data

        if self.denormalize_mean is not None and self.denormalize_std is not None:
            data = (data - self.denormalize_mean) / self.denormalize_std
        data_dict[self.key_target] = data
        data_dict[f"{self.__class__.__name__}_info"] = dict(
            implant_specs=implant_specs,
            severity=severity,
            spacing=self.spacing,
            selected_labels=selected_labels,
            angle_size=angle_size,
            cls_name=self.__class__.__name__,
        )

        return data_dict


class WireTransform(DictTransform):
    """Insert one or more curved, high-HU wires (e.g. pacemaker leads) and simulate streak artifacts.

    Same input/output contract as :class:`MetalTransform`.
    """

    def __init__(
        self,
        spacing,
        key_origin: str = "image",
        key_target: str = "image",
        max_n_specs: int = 2,
        severity: ValueOrRange = (0.1, 0.9),
        intensity: ValueOrRange = (1000, 5000),
        exclude_labels: Union[Sequence, int, None] = (0,),
        include_labels: Union[Sequence, int, None] = None,
        verbose: bool = False,
        skip_minus_one_selected_label: bool = True,
        denormalize_mean: Optional[float] = None,
        denormalize_std: Optional[float] = None,
    ):
        self.key_target = key_target
        self.key_origin = key_origin
        self.spacing = spacing
        self.max_n_specs = max_n_specs
        self.severity = severity
        self.intensity = intensity
        self.exclude_labels = exclude_labels
        self.include_labels = include_labels
        self.verbose = verbose
        self.skip_minus_one_selected_label = skip_minus_one_selected_label
        self.denormalize_mean = denormalize_mean
        self.denormalize_std = denormalize_std

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict[self.key_origin]
        data_dtype = data.dtype
        to_torch = False
        if isinstance(data, torch.Tensor):
            data = data.numpy().copy().astype(np.float32)
            to_torch = True
        seg = data_dict["segmentation"] if "segmentation" in data_dict else None
        if seg is not None and isinstance(seg, torch.Tensor):
            seg = seg.numpy().copy()
        if seg is not None and data.shape != seg.shape:
            raise ValueError(
                f"Seg and data don't have the same shape, data: {data.shape}, seg: {seg.shape}"
            )
        do_unsqueeze = False
        if len(data.shape) > 3:
            if data.shape[0] == 1:
                data = data.squeeze(0)
                if seg is not None:
                    seg = seg.squeeze(0)
                do_unsqueeze = True
            if len(data.shape) > 3:
                raise ValueError(
                    f"Input data shape is more than 3 or more than 4 while first one is 1, data shape: {data.shape}"
                )

        if self.denormalize_mean is not None and self.denormalize_std is not None:
            data = data * self.denormalize_std + self.denormalize_mean
        severity = sample_value(self.severity)
        data_shape = data.shape
        data_min, data_max = data.min(), data.max()

        n_specs = random.randint(1, max(1, self.max_n_specs))
        implant_specs = []
        selected_labels = []

        for _ in range(n_specs):
            max_slice = round(data_shape[0] * 0.2)
            z_range, center, selected_label = mask_base_position_2d(
                seg,
                data,
                max_slice,
                exclude_labels=self.exclude_labels,
                include_labels=self.include_labels,
                verbose=self.verbose,
            )

            if selected_label != -1 or not self.skip_minus_one_selected_label:
                angle_start = random.uniform(0, np.pi / 2)
                angle_end = random.uniform(
                    angle_start + np.pi / 5, angle_start + np.pi / 5 + np.pi / 2
                )
                
                # arc_radius_mm = random.randint(5, round(data_shape[0] * 0.2))   # unchanged
                in_plane_mm = min(data_shape[1] * self.spacing[1], data_shape[2] * self.spacing[2])
                arc_radius_mm = random.uniform(0.02 * in_plane_mm, 0.10 * in_plane_mm)
                
                r_row, r_col = arc_radius_mm / self.spacing[1], arc_radius_mm / self.spacing[2]
                cy = min(max(center[0], r_row), data_shape[1] - 1 - r_row)
                cx = min(max(center[1], r_col), data_shape[2] - 1 - r_col)
                center_mm = [cy * self.spacing[1], cx * self.spacing[2]]
                spec = {
                    "type": "wire",
                    "length": random.randint(
                        (z_range[1] - z_range[0]) + 1,
                        max([30, (z_range[1] - z_range[0])]) + 10,
                    ),
                    "center_mm": center_mm,
                    #     [
                    #     float(item * self.spacing[index + 1])
                    #     for index, item in enumerate(center)
                    # ],
                    "arc_radius_mm": arc_radius_mm,
                    "wire_radius_mm": random.uniform(0.01, 0.1),
                    "angle_range": (angle_start, angle_end),
                    "z_range_mm": [item * self.spacing[0] for item in z_range],
                    "intensity": sample_value(self.intensity),
                }
                if selected_label != -1:
                    selected_labels.append(selected_label)
                implant_specs.append(spec)

        angle_size = random.randint(180, 720)
        if implant_specs:
            angles = np.linspace(0, 180, angle_size, endpoint=False)
            data = simulate_artifacts_unified(
                data,
                implant_specs,
                angles=angles,
                severity=severity,
                spacing=self.spacing,
                verbose=self.verbose,
            )
            data = np.clip(data, a_min=data_min, a_max=data_max)
        if do_unsqueeze:
            data = data[None, ...]
        if to_torch:
            data = torch.from_numpy(data).to(data_dtype)
        else:
            data = data.astype(data_dtype)

        if self.denormalize_mean is not None and self.denormalize_std is not None:
            data = (data - self.denormalize_mean) / self.denormalize_std
        data_dict[self.key_target] = data

        data_dict[f"{self.__class__.__name__}_info"] = dict(
            implant_specs=implant_specs,
            severity=severity,
            spacing=self.spacing,
            selected_labels=selected_labels,
            angle_size=angle_size,
            cls_name=self.__class__.__name__,
        )
        return data_dict


class CalcificationTransform(DictTransform):
    """Insert one or more small 3D ellipsoids (e.g. calcifications) and simulate streak artifacts.

    Same input/output contract as :class:`MetalTransform`.
    """

    def __init__(
        self,
        spacing,
        key_origin: str = "image",
        key_target: str = "image",
        max_n_specs: int = 2,
        severity: ValueOrRange = (0.1, 0.9),
        intensity: ValueOrRange = (1000, 5000),
        exclude_labels: Union[Sequence, int, None] = (0,),
        include_labels: Union[Sequence, int, None] = None,
        verbose: bool = False,
        skip_minus_one_selected_label: bool = True,
        denormalize_mean: Optional[float] = None,
        denormalize_std: Optional[float] = None,
    ):
        self.key_target = key_target
        self.key_origin = key_origin
        self.spacing = spacing
        self.max_n_specs = max_n_specs
        self.severity = severity
        self.intensity = intensity
        self.exclude_labels = exclude_labels
        self.include_labels = include_labels
        self.verbose = verbose
        self.skip_minus_one_selected_label = skip_minus_one_selected_label
        self.denormalize_mean = denormalize_mean
        self.denormalize_std = denormalize_std

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict[self.key_origin]
        data_dtype = data.dtype
        to_torch = False
        if isinstance(data, torch.Tensor):
            data = data.numpy().copy().astype(np.float32)
            to_torch = True
        seg = data_dict["segmentation"] if "segmentation" in data_dict else None
        if seg is not None and isinstance(seg, torch.Tensor):
            seg = seg.numpy().copy()
        if seg is not None and data.shape != seg.shape:
            raise ValueError(
                f"Seg and data don't have the same shape, data: {data.shape}, seg: {seg.shape}"
            )
        do_unsqueeze = False
        if len(data.shape) > 3:
            if data.shape[0] == 1:
                data = data.squeeze(0)
                if seg is not None:
                    seg = seg.squeeze(0)
                do_unsqueeze = True
            if len(data.shape) > 3:
                raise ValueError(
                    f"Input data shape is more than 3 or more than 4 while first one is 1, data shape: {data.shape}"
                )
        if self.denormalize_mean is not None and self.denormalize_std is not None:
            data = data * self.denormalize_std + self.denormalize_mean
        severity = sample_value(self.severity)
        data_min, data_max = data.min(), data.max()

        n_specs = random.randint(1, max(1, self.max_n_specs))
        implant_specs = []
        selected_labels = []

        for _ in range(n_specs):
            center, selected_label = mask_base_position_3d(
                seg,
                data,
                exclude_labels=self.exclude_labels,
                include_labels=self.include_labels,
                verbose=self.verbose,
            )
            if selected_label != -1 or not self.skip_minus_one_selected_label:
                spec = {
                    "type": "3d",
                    "center_mm": [
                        float(item * self.spacing[index])
                        for index, item in enumerate(center)
                    ],
                    "radius_mm": (
                        random.randint(2, 10),
                        random.uniform(0.5, 2),
                        random.uniform(0.5, 2),
                    ),
                    "intensity": sample_value(self.intensity),
                }
                if selected_label != -1:
                    selected_labels.append(selected_label)
                implant_specs.append(spec)

        angle_size = random.randint(180, 720)
        if implant_specs:
            angles = np.linspace(0, 180, angle_size, endpoint=False)
            data = simulate_artifacts_unified(
                data,
                implant_specs,
                angles=angles,
                severity=severity,
                spacing=self.spacing,
                verbose=self.verbose,
            )
            data = np.clip(data, a_min=data_min, a_max=data_max)
        if do_unsqueeze:
            data = data[None, ...]
        if to_torch:
            data = torch.from_numpy(data).to(data_dtype)
        else:
            data = data.astype(data_dtype)

        if self.denormalize_mean is not None and self.denormalize_std is not None:
            data = (data - self.denormalize_mean) / self.denormalize_std
        data_dict[self.key_target] = data

        data_dict[f"{self.__class__.__name__}_info"] = dict(
            implant_specs=implant_specs,
            severity=severity,
            spacing=self.spacing,
            selected_labels=selected_labels,
            angle_size=angle_size,
            cls_name=self.__class__.__name__,
        )
        return data_dict


class RandomArtifactTransform(DictTransform):
    """Randomly apply one of :class:`MetalTransform`, :class:`WireTransform`, or :class:`CalcificationTransform`."""

    def __init__(
        self,
        spacing,
        key_origin: str = "image",
        key_target: str = "image",
        max_n_specs: int = 2,
        severity: ValueOrRange = (0.1, 0.9),
        intensity: ValueOrRange = (1000, 5000),
        exclude_labels: Union[Sequence, int, None] = (0,),
        include_labels: Union[Sequence, int, None] = None,
        verbose: bool = False,
        skip_minus_one_selected_label: bool = True,
        exclude_class: Optional[Tuple[str, ...]] = None,
        denormalize_mean: Optional[float] = None,
        denormalize_std: Optional[float] = None,
    ):
        self.exclude_class = exclude_class

        self.calcification = CalcificationTransform(
            spacing=spacing,
            key_origin=key_origin,
            key_target=key_target,
            max_n_specs=max_n_specs,
            severity=severity,
            intensity=intensity,
            exclude_labels=exclude_labels,
            include_labels=include_labels,
            verbose=verbose,
            skip_minus_one_selected_label=skip_minus_one_selected_label,
            denormalize_mean=denormalize_mean,
            denormalize_std=denormalize_std,
        )
        self.metal = MetalTransform(
            spacing=spacing,
            key_origin=key_origin,
            key_target=key_target,
            max_n_specs=max_n_specs,
            severity=severity,
            intensity=intensity,
            exclude_labels=exclude_labels,
            include_labels=include_labels,
            verbose=verbose,
            skip_minus_one_selected_label=skip_minus_one_selected_label,
            denormalize_mean=denormalize_mean,
            denormalize_std=denormalize_std,
        )
        self.wire = WireTransform(
            spacing=spacing,
            key_origin=key_origin,
            key_target=key_target,
            max_n_specs=max_n_specs,
            severity=severity,
            intensity=intensity,
            exclude_labels=exclude_labels,
            include_labels=include_labels,
            verbose=verbose,
            skip_minus_one_selected_label=skip_minus_one_selected_label,
            denormalize_mean=denormalize_mean,
            denormalize_std=denormalize_std,
        )

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        name_func_dict = dict(
            wire=self.wire, metal=self.metal, calcification=self.calcification
        )
        if self.exclude_class:
            for item in self.exclude_class:
                name_func_dict.pop(item)
        return random.choice(list(name_func_dict.values()))(**data_dict)
