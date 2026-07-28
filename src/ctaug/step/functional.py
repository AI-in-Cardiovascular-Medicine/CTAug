
import random
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from ctaug._resize import resize_volume


def step_core(img: np.ndarray, cutoff_index: int, cut_off_position: int,
                        intensity_value: float) -> np.ndarray:
    """Add a constant intensity offset to one side of a volume along ``cutoff_index``.

    Mimics the intensity discontinuity seen at step-and-shoot (gated) acquisition boundaries.
    """
    image = img.copy()
    add_to_tail = random.random() < 0.5
    if cutoff_index == 0:
        if add_to_tail:
            image[cut_off_position:, ...] += intensity_value
        else:
            image[:cut_off_position, ...] += intensity_value
    elif cutoff_index == 1:
        if add_to_tail:
            image[:, cut_off_position:, ...] += intensity_value
        else:
            image[:, :cut_off_position, ...] += intensity_value
    elif cutoff_index == 2 or cutoff_index == -1:
        if add_to_tail:
            image[:, :, cut_off_position:] += intensity_value
        else:
            image[:, :, :cut_off_position] += intensity_value
    else:
        raise ValueError("Only 3D images are supported!")
    return image


def motion_core(
        image: np.ndarray,
        cutoff_index: int,
        move_index: int,
        cropped_value: str,
        cut_off_position: int,
        img_size: Tuple[int, ...],
        mean_value: float,
        move_value: Union[int, float],
        move_left: Optional[bool] = None,
        input_type: str = "image") -> np.ndarray:
    """Shift the region of ``image`` beyond ``cut_off_position`` (along ``cutoff_index``) by
    ``move_value`` voxels along ``move_index``, simulating a sudden patient/gantry motion.

    ``cropped_value`` controls how the region vacated by the shift is filled: ``"zero"``,
    ``"mean"`` (fill with ``mean_value``), or ``"cut"``/``"cut_resize"`` (crop the volume and,
    for ``"cut_resize"``, resize it back to ``img_size``).
    """
    image = image.copy()
    move_left = move_left if move_left is not None else random.random() < 0.5
    val = 0 if cropped_value == "zero" else mean_value

    if cutoff_index == -1 or cutoff_index == 2:
        if move_index == 1:
            if move_left:
                image[:, :image.shape[1] - move_value, cut_off_position:] = image[:, move_value:, cut_off_position:]
            else:
                image[:, move_value:, cut_off_position:] = image[:, :image.shape[1] - move_value, cut_off_position:]
            if cropped_value in ("zero", "mean"):
                if move_left:
                    image[:, image.shape[1] - move_value:, cut_off_position:] = val
                else:
                    image[:, :move_value, cut_off_position:] = val
            else:
                image = image[:, :image.shape[1] - move_value, :] if move_left else image[:, :move_value, :]
        elif move_index == 0:
            if move_left:
                image[:image.shape[0] - move_value, :, cut_off_position:] = image[move_value:, :, cut_off_position:]
            else:
                image[move_value:, :, cut_off_position:] = image[:image.shape[0] - move_value, :, cut_off_position:]
            if cropped_value in ("zero", "mean"):
                if move_left:
                    image[image.shape[0] - move_value:, :, cut_off_position:] = val
                else:
                    image[:move_value, :, cut_off_position:] = val
            else:
                image = image[:image.shape[0] - move_value, :, :] if move_left else image[move_value:, :, :]
        else:
            raise ValueError(f"move_index: {move_index} is not valid!")

    elif cutoff_index == 1:
        if move_index == 0:
            if move_left:
                image[:image.shape[0] - move_value, cut_off_position:, :] = image[move_value:, cut_off_position:, :]
            else:
                image[move_value:, cut_off_position:, :] = image[:image.shape[0] - move_value, cut_off_position:, :]
            if cropped_value in ("zero", "mean"):
                if move_left:
                    image[image.shape[0] - move_value:, cut_off_position:, :] = val
                else:
                    image[:move_value, cut_off_position:, :] = val
            else:
                image = image[:image.shape[0] - move_value, :, :] if move_left else image[move_value:, :, :]
        elif move_index == 2:
            if move_left:
                image[:, cut_off_position:, :image.shape[2] - move_value] = image[:, cut_off_position:, move_value:]
            else:
                image[:, cut_off_position:, move_value:] = image[:, cut_off_position:, :image.shape[2] - move_value]
            if cropped_value in ("zero", "mean"):
                if move_left:
                    image[:, cut_off_position:, image.shape[2] - move_value:] = val
                else:
                    image[:, cut_off_position:, :move_value] = val
            else:
                image = image[:, :, :image.shape[2] - move_value] if move_left else image[:, :, move_value:]
        else:
            raise ValueError(f"move_index: {move_index} is not valid!")

    elif cutoff_index == 0:
        if move_index == 1:
            if move_left:
                image[cut_off_position:, :image.shape[1] - move_value, :] = image[cut_off_position:, move_value:, :]
            else:
                image[cut_off_position:, move_value:, :] = image[cut_off_position:, :image.shape[1] - move_value, :]
            if cropped_value in ("zero", "mean"):
                if move_left:
                    image[cut_off_position:, image.shape[1] - move_value:, :] = val
                else:
                    image[cut_off_position:, :move_value, :] = val
            else:
                image = image[:, :image.shape[1] - move_value, :] if move_left else image[:, move_value:, :]
        elif move_index == 2:
            if move_left:
                image[cut_off_position:, :, :image.shape[2] - move_value] = image[cut_off_position:, :, move_value:]
            else:
                image[cut_off_position:, :, move_value:] = image[cut_off_position:, :, :image.shape[2] - move_value]
            if cropped_value in ("zero", "mean"):
                if move_left:
                    image[cut_off_position:, :, image.shape[2] - move_value:] = val
                else:
                    image[cut_off_position:, :, :move_value] = val
            else:
                image = image[:, :, :image.shape[2] - move_value] if move_left else image[:, :, move_value:]
        else:
            raise ValueError(f"move_index: {move_index} is not valid!")
    else:
        raise ValueError(f"cutoff_index: {cutoff_index} is not valid!")

    if cropped_value == "cut_resize":
        image = resize_volume(image, img_size, input_type=input_type)
    return image


class StepAugmentation:
    """Add a step-wise intensity discontinuity to a 3D volume, simulating gated acquisition seams."""

    def __init__(self, cutoff_index: int = -1,
                cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                normalize: bool = True):
        """
        :param cutoff_index: axis along which the intensity step is introduced.
        :param cut_off_pixel_value_weight: the mean pixel value is calculated then a value between the
            given bounds is randomly chosen and added to one side of the volume.
        :param normalize: whether to rescale the output back to the input's min/max range.
        """
        assert 0 <= cut_off_pixel_value_weight[0]
        self.cutoff_index = cutoff_index
        self.cut_off_pixel_value_weight = cut_off_pixel_value_weight
        self.normalize = normalize

    def __call__(self, data: np.ndarray):
        image: np.ndarray = data.copy().astype(np.float32)
        if self.normalize:
            min_value, max_value = np.min(image), np.max(image)

        cut_off_position = random.randint(0, image.shape[self.cutoff_index] - 1)
        mean_pixel_value = np.mean(image)
        intensity_value = random.uniform(self.cut_off_pixel_value_weight[0] * mean_pixel_value,
                                         self.cut_off_pixel_value_weight[1] * mean_pixel_value)
        intensity_value = intensity_value if random.random() < 0.5 else -intensity_value

        image = step_core(img=image, 
                          cutoff_index=self.cutoff_index,
                          cut_off_position=cut_off_position,
                          intensity_value=intensity_value)
        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = image * (max_value - min_value) + min_value

        info = dict(cut_off_position=cut_off_position, normalize=self.normalize, 
                    intensity_value=float(intensity_value), cutoff_index=self.cutoff_index)
        return image, info


class MotionAugmentation:
    """Shift part of a 3D volume (and its segmentation) to simulate a sudden motion event."""

    def __init__(self,
                motion_move_position: Union[Tuple[float, float], List[float]] = (0.05, 0.2),
                motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.05),
                normalize: bool = True):
        """
        :param motion_move_position: fraction range (of the cutoff axis extent) from which the
            cutoff position is sampled.
        :param motion_move_range: fraction range (of the move axis extent) by which the shifted
            region is displaced.
        :param normalize: whether to rescale the output back to the input's min/max range.
        """
        assert 0 <= motion_move_range[0]
        assert 0 <= motion_move_position[0]
        self.motion_move_range = motion_move_range
        self.motion_move_position = motion_move_position
        self.normalize = normalize

    def __call__(self, data: np.ndarray, seg_data: np.ndarray,
                cropped_value: Literal["zero", "mean", "cut", "cut_resize"] = None,
                cutoff_index: Optional[int] = None, move_index: Optional[int] = None):
        cropped_value = cropped_value or random.choice(["zero", "mean", "cut_resize"])
        cutoff_index = cutoff_index if cutoff_index is not None else random.choice([1, 2])
        move_index = move_index if move_index is not None else random.choice(list({0, 1, 2} - {cutoff_index}))

        image: np.ndarray = data.copy().astype(np.float32)
        seg: np.ndarray = seg_data.copy().astype(np.float32)

        if self.normalize:
            image_min_value, image_max_value = np.min(image), np.max(image)

        image_mean_value = float(np.mean(image))
        seg_mean_value = 0
        img_size = image.shape

        start = round(image.shape[move_index] * self.motion_move_range[0])
        end = round(image.shape[move_index] * self.motion_move_range[1]) - 1
        end = (start + 1) if end <= start else end
        move_value = random.randint(start, end)

        start = round(image.shape[cutoff_index] * self.motion_move_position[0])
        end = round(image.shape[cutoff_index] * self.motion_move_position[1])
        end = (start + 1) if end <= start else end
        cut_off_position = random.randint(start, end)
        cut_off_position = cut_off_position if random.random() < 0.5 else (image.shape[cutoff_index] - cut_off_position)
        move_left = random.random() < 0.5

        image = motion_core(image, cutoff_index, move_index, cropped_value, cut_off_position, img_size,
                            image_mean_value, move_value, move_left=move_left, input_type="image")
        seg = motion_core(seg, cutoff_index, move_index, cropped_value, cut_off_position, img_size,
                          seg_mean_value, move_value, move_left=move_left, input_type="label")

        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = image * (image_max_value - image_min_value) + image_min_value

        info = dict(move_index=move_index, cutoff_index=cutoff_index, cropped_value=cropped_value,
                    cut_off_position=cut_off_position, move_value=move_value, move_left=move_left)
        return image, seg, info


class StepMotionAugmentation:
    """Combine :class:`StepAugmentation` and :class:`MotionAugmentation` in a single pass."""

    def __init__(self,
                cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                motion_move_range: Union[Tuple[float, float], List[float]] = (0.02, 0.1),
                normalize: bool = True):
        """
        :param cut_off_pixel_value_weight: the mean pixel value is calculated then a value between the
            given bounds is randomly chosen and added to one side of the volume.
        :param motion_move_range: fraction range (of the move axis extent) by which the shifted
            region is displaced.
        :param normalize: whether to rescale the output back to the input's min/max range.
        """
        assert 0 <= cut_off_pixel_value_weight[0]
        assert 0 <= motion_move_range[0]
        self.cut_off_pixel_value_weight = cut_off_pixel_value_weight
        self.motion_move_range = motion_move_range
        self.normalize = normalize

    def __call__(self, data: np.ndarray, seg_data: Optional[np.ndarray],
                cropped_value: Literal["zero", "mean", "cut", "cut_resize"] = None,
                cutoff_index: Optional[int] = None, move_index: Optional[int] = None):
        cropped_value = cropped_value or random.choice(["zero", "mean", "cut_resize"])
        cutoff_index = cutoff_index if cutoff_index is not None else 0
        move_index = move_index if move_index is not None else random.choice(list({0, 1, 2} - {cutoff_index}))

        image: np.ndarray = data.copy().astype(np.float32)
        seg = seg_data.copy() if seg_data is not None else None

        if self.normalize:
            image_min_value, image_max_value = np.min(image), np.max(image)

        image_mean_value = float(np.mean(image))
        seg_mean_value = 0
        cut_off_position = random.randint(0, image.shape[cutoff_index] - 1)
        img_size = image.shape

        start = round(image.shape[move_index] * self.motion_move_range[0])
        end = round(image.shape[move_index] * self.motion_move_range[1]) - 1
        end = (start + 1) if end <= start else end
        move_value = random.randint(start, end)

        intensity_value = random.uniform(self.cut_off_pixel_value_weight[0] * image_mean_value,
                                         self.cut_off_pixel_value_weight[1] * image_mean_value)
        intensity_value = intensity_value if random.random() < 0.5 else -intensity_value
        move_left = random.random() < 0.5

        image = step_core(img=image, cutoff_index=cutoff_index, cut_off_position=cut_off_position,
                                    intensity_value=intensity_value)
        image = motion_core(image, cutoff_index, move_index, cropped_value, cut_off_position, img_size,
                            image_mean_value, move_value, move_left=move_left, input_type="image")
        if seg is not None:
            seg = motion_core(seg, cutoff_index, move_index, cropped_value, cut_off_position, img_size,
                              seg_mean_value, move_value, move_left=move_left, input_type="label")

        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
            image = image * (image_max_value - image_min_value) + image_min_value

        info = dict(move_index=move_index, cutoff_index=cutoff_index, cropped_value=cropped_value,
                    cut_off_position=cut_off_position, move_value=move_value, move_left=move_left)
        return image, seg, info
