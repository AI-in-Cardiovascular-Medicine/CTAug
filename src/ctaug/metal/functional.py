import random
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import iradon, radon
from skimage.transform import resize as sk_resize


def insert_2d_shape_mm(slice_2d: np.ndarray, center_px, radius_mm, spacing, shape: str = "circle",
                       intensity: float = 3000):
    """Burn a circle/ellipse/rectangle of the given HU intensity into a 2D slice."""
    rr, cc = np.ogrid[:slice_2d.shape[0], :slice_2d.shape[1]]
    r_center, c_center = center_px
    if isinstance(radius_mm, (int, float)):
        r_radius, c_radius = radius_mm / spacing[0], radius_mm / spacing[1]
    else:
        r_radius, c_radius = radius_mm[0] / spacing[0], radius_mm[1] / spacing[1]

    if shape == "circle":
        mask = (rr - r_center) ** 2 + (cc - c_center) ** 2 <= r_radius ** 2
    elif shape == "ellipse":
        mask = ((rr - r_center) ** 2) / r_radius ** 2 + ((cc - c_center) ** 2) / c_radius ** 2 <= 1
    elif shape == "rectangle":
        mask = (np.abs(rr - r_center) <= r_radius) & (np.abs(cc - c_center) <= c_radius)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    slice_2d[mask] = intensity
    return slice_2d, mask.astype(np.uint8)


def create_3d_ellipsoid_mask_mm(shape, center_mm, radius_mm, spacing) -> np.ndarray:
    """Boolean ellipsoid mask for a volume, with center/radius given in millimeters."""
    center_vox = np.array(center_mm) / np.array(spacing)
    radius_vox = np.array(radius_mm) / np.array(spacing)
    zz, xx, yy = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cx, cy = center_vox
    rz, rx, ry = radius_vox
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 + ((zz - cz) / rz) ** 2 <= 1
    return mask.astype(np.uint8)


def create_3d_ellipsoid_mask(shape, center, radius) -> np.ndarray:
    """Boolean ellipsoid mask for a volume, with center/radius given in voxels."""
    center_vox = np.array(center)
    radius_vox = np.array(radius)
    zz, xx, yy = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cx, cy = center_vox
    rz, rx, ry = radius_vox
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 + ((zz - cz) / rz) ** 2 <= 1
    return mask.astype(np.uint8)


def generate_moon_curve_path(length, center, radius, angle_range, z_range, spacing) -> np.ndarray:
    """Arc-shaped voxel path used to draw a curved wire (e.g. a pacemaker lead)."""
    theta = np.linspace(angle_range[0], angle_range[1], length)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.linspace(z_range[0], z_range[1], length)
    x = x / spacing[1]
    y = y / spacing[2]
    z = z / spacing[0]
    return np.stack([z, x, y], axis=1)


def draw_wire_in_volume(shape, path_vox, wire_radius_mm, intensity, spacing) -> np.ndarray:
    """Rasterize a voxel path into a smoothed, high-intensity wire volume."""
    vol = np.zeros(shape, dtype=np.float32)
    for pt in path_vox.astype(int):
        z, x, y = pt
        if 0 <= y < shape[2] and 0 <= x < shape[1] and 0 <= z < shape[0]:
            vol[z, x, y] = intensity
    sigma_vox = np.array(wire_radius_mm) / np.array(spacing)
    vol = gaussian_filter(vol, sigma=sigma_vox)
    vol[vol > 0] = intensity
    return vol


def simulate_artifact(slice_2d: np.ndarray, mask_2d: np.ndarray, angles, severity: float = 0.1) -> np.ndarray:
    """Simulate radiating streak artifacts by attenuating the sinogram under the metal mask.

    The slice is projected with the Radon transform, the sinogram rows explained by the
    metal region are dampened by ``severity``, and the result is reconstructed with the
    inverse Radon transform, mimicking the streaking seen around real metal implants.
    """
    input_shape = slice_2d.shape
    input_size = min(input_shape)
    if input_shape != (input_size, input_size):
        mask_2d = sk_resize(mask_2d.astype(np.uint8), (input_size, input_size), order=0,
                            preserve_range=True, anti_aliasing=False).astype(np.bool_)
        slice_2d = sk_resize(slice_2d, (input_size, input_size), order=1,
                             preserve_range=True, anti_aliasing=True)

    metal_only = np.where(mask_2d, slice_2d, 0)
    R = radon(slice_2d, theta=angles, circle=False)
    R_metal = radon(metal_only, theta=angles, circle=False)
    R_weight = R_metal / np.max(R) if np.max(R) > 0 else R_metal
    R_weight = np.clip(R_weight, 0, 1)
    sino_new = R * (1 - severity * R_weight)
    output = iradon(sino_new, theta=angles, filter_name="cosine", circle=False,
                    output_size=input_size).astype(np.float32)
    output = sk_resize(output, input_shape, order=1, preserve_range=True,
                       anti_aliasing=True).astype(np.float32)
    return output


def process_slice(z: int, modified_data: np.ndarray, final_mask: np.ndarray, severity: float, angles) -> np.ndarray:
    slice_2d = modified_data[z, :, :]
    mask_2d = final_mask[z, :, :]
    if np.any(mask_2d):
        return simulate_artifact(slice_2d, mask_2d, angles=angles, severity=severity)
    return slice_2d.astype(np.float32)


def mask_base_position_2d(mask: Optional[np.ndarray], data: np.ndarray, max_slice: int,
                          exclude_labels: Union[Sequence, int, None] = (0,),
                          include_labels: Union[Sequence, int, None] = None, 
                          verbose: bool = False):
    """Pick a random (z-range, in-plane center) anchored on a segmentation label, when available.

    Falls back to a uniformly random position (with ``selected_label`` set to ``-1``) if no
    mask is given, the mask has no eligible labels, or a candidate region turns out empty.
    """
    success = False
    center = None
    z_start = z_end = None
    selected_label = -1
    labels = []
    if mask is not None:
        crop_labels = set(np.unique(mask))
        if include_labels is not None:
            labels = (include_labels,) if isinstance(include_labels, int) else list(include_labels)
        elif exclude_labels is not None:
            exclude_labels = (exclude_labels,) if isinstance(exclude_labels, int) else exclude_labels
            labels = list(crop_labels - set(exclude_labels))
        else:
            # no filter given: every label present in the mask is eligible
            labels = list(crop_labels)
        if labels:
            for _ in range(5):
                try:
                    selected_label = random.choice(labels)
                    zs, xs, ys = np.where(mask == selected_label)
                    min_z, max_z = min(zs), max(zs)
                    z_start = random.randint(min_z, max_z - 1)
                    z_end = random.randint(z_start + 1, min(max_z, z_start + 1 + max_slice))

                    _, xs, ys = np.where(mask[z_start:z_end, ...] == selected_label)
                    centers = [(x, y) for x, y in zip(xs, ys)]
                    if centers:
                        center = random.choice(centers)
                        success = True
                        break
                except Exception as e:
                    if verbose:
                        warnings.warn(f"Error in mask_base_position_2d: {crop_labels=} -> {e=}")
                    continue
    if not success:
        z_start = random.randint(0, data.shape[0] - 1 - max_slice)
        z_end = random.randint(z_start + 1, min(z_start + 1 + max_slice, data.shape[0]))
        center = (random.randint(0, data.shape[1] - 1), random.randint(0, data.shape[2] - 1))
        selected_label = -1
        if verbose and mask is not None:
            warnings.warn("No eligible segmentation label found in mask_base_position_2d, using a random position")
    return (z_start, z_end), center, selected_label


def mask_base_position_3d(mask: Optional[np.ndarray], data: np.ndarray,
                          exclude_labels: Union[Sequence, int, None] = (0,), include_labels: Union[Sequence, int, None] = None, 
                          verbose: bool = False):
    """Pick a random 3D voxel anchored on a segmentation label, when available."""
    success = False
    center = None
    selected_label = -1
    labels = []
    if mask is not None:
        crop_labels = set(np.unique(mask))
        if include_labels is not None:
            labels = (include_labels,) if isinstance(include_labels, int) else list(include_labels)
        elif exclude_labels is not None:
            exclude_labels = (exclude_labels,) if isinstance(exclude_labels, int) else exclude_labels
            labels = list(crop_labels - set(exclude_labels))
        else:
            # no filter given: every label present in the mask is eligible
            labels = list(crop_labels)
        if labels:
            try:
                selected_label = random.choice(labels)
                zs, xs, ys = np.where(mask == selected_label)
                centers = [(z, x, y) for z, x, y in zip(zs, xs, ys)]
                center = random.choice(centers)
                success = True
            except Exception as e:
                if verbose:
                    warnings.warn(f"Error in mask_base_position_3d: {crop_labels=} -> {e=}")
    if not success:
        selected_label = -1
        center = [random.randint(0, data.shape[0] - 1),
                  random.randint(0, data.shape[1] - 1),
                  random.randint(0, data.shape[2] - 1)]
        if verbose and mask is not None:
            warnings.warn("No eligible segmentation label found in mask_base_position_3d, using a random position")
    return center, selected_label


def simulate_artifacts_unified(data: np.ndarray, implant_specs: List[Dict], spacing, angles,
                               severity: float = 0.05, smooth_sigma: float = 1.0,
                               verbose: bool = True) -> np.ndarray:
    """Insert one or more implant specs into ``data`` and simulate the resulting streak artifacts.

    ``implant_specs`` is a list of dicts, each with a ``"type"`` of ``"2d"``, ``"3d"``, or
    ``"wire"``; see :mod:`ctaug.metal.transforms` for how each type's parameters are sampled.
    """
    shape = data.shape
    final_mask = np.zeros_like(data, dtype=bool)
    modified_data = data.copy().astype(np.float32)

    for imp in implant_specs:
        if imp["type"] == "2d":
            for z in imp["slices"]:
                modified_data[z, :, :], mask = insert_2d_shape_mm(
                    modified_data[z, :, :],
                    center_px=imp["center_px"],
                    radius_mm=imp["radius_mm"],
                    spacing=spacing[1:],
                    shape=imp["shape"],
                    intensity=imp.get("intensity", 3000),
                )
                final_mask[z, :, :] |= mask.astype(bool)

        elif imp["type"] == "3d":
            mask = create_3d_ellipsoid_mask_mm(
                shape,
                center_mm=imp["center_mm"],
                radius_mm=imp["radius_mm"],
                spacing=spacing,
            )
            if smooth_sigma > 0:
                mask = gaussian_filter(mask.astype(float), sigma=smooth_sigma) > 0.2
            modified_data[mask] = imp["intensity"]
            final_mask |= mask

        elif imp["type"] == "wire":
            path = generate_moon_curve_path(
                length=imp["length"],
                center=imp["center_mm"],
                radius=imp["arc_radius_mm"],
                angle_range=imp["angle_range"],
                z_range=imp["z_range_mm"],
                spacing=spacing,
            )
            mask = draw_wire_in_volume(
                shape,
                path,
                wire_radius_mm=imp["wire_radius_mm"],
                intensity=imp["intensity"],
                spacing=spacing,
            )
            modified_data[mask > 0] = imp["intensity"]
            final_mask |= (mask > 0)
        else:
            raise ValueError(f"imp_type: {imp['type']} is not supported!")

    if verbose and not final_mask.any():
        warnings.warn(f"Nothing is in the mask for implant_specs={implant_specs}")

    artifact_volume = np.stack(
        [process_slice(z, modified_data, final_mask, severity=severity, angles=angles) for z in range(shape[0])],
        axis=0,
    )
    return artifact_volume
