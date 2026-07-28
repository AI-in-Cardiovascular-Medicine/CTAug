
from typing import List, Optional, Tuple, Union

from monai.config import KeysCollection

from ctaug.monai._adapter import CTAugDictAdapter
from ctaug.step.transforms import (
    MotionTransform,
    StepMotionTransform,
    StepTransform,
)


class Stepd(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.step.transforms.StepTransform`.

    Adds a step-wise intensity discontinuity, simulating gated acquisition seams.

    :param keys: image key(s) to augment. No segmentation is required or modified.
    :param cutoff_index: axis along which the intensity step is introduced.
    :param cut_off_pixel_value_weight: the mean pixel value is calculated then a value between
        the given bounds is randomly chosen and added to one side of the volume.
    :param normalize: whether to rescale the output back to the input's min/max range.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection,
                cutoff_index: int = -1,
                cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                normalize: bool = True,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = StepTransform(cutoff_index=cutoff_index,
                             cut_off_pixel_value_weight=cut_off_pixel_value_weight,
                             normalize=normalize)
        super().__init__(core, keys, label_key=None, prob=prob,
                         allow_missing_keys=allow_missing_keys)


class Motiond(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.step.transforms.MotionTransform`.

    Shifts part of the volume to simulate a sudden motion event.

    :param keys: image key(s) to augment.
    :param label_key: segmentation key, required — it is shifted alongside the image.
    :param motion_move_position: fraction range (of the cutoff axis extent) from which the
        cutoff position is sampled.
    :param motion_move_range: fraction range (of the move axis extent) by which the shifted
        region is displaced.
    :param normalize: whether to rescale the output back to the input's min/max range.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection, label_key: str = "label",
                motion_move_position: Union[Tuple[float, float], List[float]] = (0.05, 0.2),
                motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.015),
                normalize: bool = True,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = MotionTransform(motion_move_position=motion_move_position,
                               motion_move_range=motion_move_range,
                               normalize=normalize)
        super().__init__(core, keys, label_key=label_key, require_label=True, prob=prob,
                         allow_missing_keys=allow_missing_keys)


class StepMotiond(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.step.transforms.StepMotionTransform`.

    Combines a step-wise intensity discontinuity with a motion shift at the same boundary.
    This is the step-and-shoot augmentation used in the paper.

    :param keys: image key(s) to augment.
    :param label_key: optional segmentation key; shifted alongside the image when present.
    :param cut_off_pixel_value_weight: the mean pixel value is calculated then a value between
        the given bounds is randomly chosen and added to one side of the volume.
    :param motion_move_range: fraction range (of the move axis extent) by which the shifted
        region is displaced.
    :param normalize: whether to rescale the output back to the input's min/max range.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection, label_key: Optional[str] = "label",
                cut_off_pixel_value_weight: Union[Tuple[float, float], List[float]] = (0.2, 0.6),
                motion_move_range: Union[Tuple[float, float], List[float]] = (0.01, 0.025),
                normalize: bool = True,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = StepMotionTransform(cut_off_pixel_value_weight=cut_off_pixel_value_weight,
                                   motion_move_range=motion_move_range,
                                   normalize=normalize)
        super().__init__(core, keys, label_key=label_key, require_label=False,
                         prob=prob, allow_missing_keys=allow_missing_keys)
