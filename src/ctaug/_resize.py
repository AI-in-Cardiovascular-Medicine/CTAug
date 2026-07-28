from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def resize_volume(volume: np.ndarray, target_size: Sequence[int], input_type: str) -> np.ndarray:
    """Resize a 3D array with torch, using an interpolation mode appropriate for images or labels.

    :param volume: 3D array (Z, Y, X).
    :param target_size: target (Z, Y, X) shape.
    :param input_type: "image" for intensity data (trilinear) or "label" for segmentation masks (nearest).
    """
    if input_type == "image":
        mode, kwargs = "trilinear", {"align_corners": False}
    elif input_type == "label":
        mode, kwargs = "nearest-exact", {}
    else:
        raise ValueError(f"input_type: {input_type} is not correct!")

    input_dtype = volume.dtype
    tensor = torch.as_tensor(volume, dtype=torch.float32)[None, None, ...]
    resized = F.interpolate(tensor, size=tuple(int(s) for s in target_size), mode=mode, **kwargs)
    return resized[0, 0].numpy().astype(input_dtype)
