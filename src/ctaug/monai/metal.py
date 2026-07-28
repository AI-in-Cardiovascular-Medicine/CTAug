
from typing import Optional, Sequence, Tuple, Union

from monai.config import KeysCollection

from ctaug.metal.transforms import (
    CalcificationTransform,
    ValueOrRange,
    MetalTransform,
    RandomArtifactTransform,
    WireTransform,
)
from ctaug.monai._adapter import CTAugDictAdapter


class Metald(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.metal.transforms.MetalTransform`.

    Inserts 2D metal implants (circle/ellipse/rectangle) and simulates streak artifacts.

    :param keys: image key(s) to augment.
    :param spacing: voxel spacing (Z, Y, X), in mm.
    :param label_key: segmentation key used to anchor implant placement; also written back.
    :param max_n_specs: upper bound on the number of implants inserted per call.
    :param severity: streak severity -- a scalar, or a ``(low, high)`` range sampled per call.
    :param intensity: implant HU value -- a scalar, or a ``(low, high)`` range sampled per implant.
    :param exclude_labels: labels that may not anchor an implant.
    :param include_labels: if given, the only labels that may anchor an implant.
    :param verbose: whether to warn when placement falls back to a random position.
    :param skip_minus_one_selected_label: skip implants that could not be anchored on a label.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection, spacing, label_key: str = "label", max_n_specs: int = 2,
                severity: ValueOrRange = (0.1, 0.9), intensity: ValueOrRange = (1000, 5000),
                exclude_labels: Union[Sequence, int, None] = (0,),
                include_labels: Union[Sequence, int, None] = None, verbose: bool = False,
                skip_minus_one_selected_label: bool = True,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = MetalTransform(spacing=spacing, max_n_specs=max_n_specs, severity=severity,
                              intensity=intensity,
                              exclude_labels=exclude_labels, include_labels=include_labels,
                              verbose=verbose,
                              skip_minus_one_selected_label=skip_minus_one_selected_label)
        super().__init__(core, keys, label_key=label_key, prob=prob, allow_missing_keys=allow_missing_keys)


class Wired(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.metal.transforms.WireTransform`.

    Inserts curved, high-HU wires (e.g. pacemaker leads) and simulates streak artifacts.

    :param keys: image key(s) to augment.
    :param spacing: voxel spacing (Z, Y, X), in mm.
    :param label_key: segmentation key used to anchor wire placement; also written back.
    :param max_n_specs: upper bound on the number of wires inserted per call.
    :param severity: streak severity -- a scalar, or a ``(low, high)`` range sampled per call.
    :param intensity: implant HU value -- a scalar, or a ``(low, high)`` range sampled per implant.
    :param exclude_labels: labels that may not anchor a wire.
    :param include_labels: if given, the only labels that may anchor a wire.
    :param verbose: whether to warn when placement falls back to a random position.
    :param skip_minus_one_selected_label: skip wires that could not be anchored on a label.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection, spacing, label_key: str = "label", max_n_specs: int = 2,
                severity: ValueOrRange = (0.1, 0.9), intensity: ValueOrRange = (1000, 5000),
                exclude_labels: Union[Sequence, int, None] = (0,),
                include_labels: Union[Sequence, int, None] = None, verbose: bool = False,
                skip_minus_one_selected_label: bool = True,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = WireTransform(spacing=spacing, max_n_specs=max_n_specs, severity=severity,
                             intensity=intensity,
                             exclude_labels=exclude_labels, include_labels=include_labels,
                             verbose=verbose,
                             skip_minus_one_selected_label=skip_minus_one_selected_label)
        super().__init__(core, keys, label_key=label_key, prob=prob, allow_missing_keys=allow_missing_keys)


class Calcificationd(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.metal.transforms.CalcificationTransform`.

    Inserts small 3D ellipsoids (e.g. calcifications) and simulates streak artifacts.

    :param keys: image key(s) to augment.
    :param spacing: voxel spacing (Z, Y, X), in mm.
    :param label_key: segmentation key used to anchor deposit placement; also written back.
    :param max_n_specs: upper bound on the number of deposits inserted per call.
    :param severity: streak severity -- a scalar, or a ``(low, high)`` range sampled per call.
    :param intensity: implant HU value -- a scalar, or a ``(low, high)`` range sampled per implant.
    :param exclude_labels: labels that may not anchor a deposit.
    :param include_labels: if given, the only labels that may anchor a deposit.
    :param verbose: whether to warn when placement falls back to a random position.
    :param skip_minus_one_selected_label: skip deposits that could not be anchored on a label.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection, spacing, label_key: str = "label", max_n_specs: int = 2,
                severity: ValueOrRange = (0.1, 0.9), intensity: ValueOrRange = (1000, 5000),
                exclude_labels: Union[Sequence, int, None] = (0,),
                include_labels: Union[Sequence, int, None] = None, verbose: bool = False,
                skip_minus_one_selected_label: bool = True,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = CalcificationTransform(spacing=spacing, max_n_specs=max_n_specs, severity=severity,
                                      intensity=intensity,
                                      exclude_labels=exclude_labels, include_labels=include_labels,
                                      verbose=verbose,
                                      skip_minus_one_selected_label=skip_minus_one_selected_label)
        super().__init__(core, keys, label_key=label_key, prob=prob, allow_missing_keys=allow_missing_keys)


class RandomArtifactd(CTAugDictAdapter):
    """MONAI dict transform wrapping :class:`~ctaug.metal.transforms.RandomArtifactTransform`.

    Applies one of the metal, wire, or calcification artifacts at random.

    :param keys: image key(s) to augment.
    :param spacing: voxel spacing (Z, Y, X), in mm.
    :param label_key: segmentation key used to anchor implant placement; also written back.
    :param max_n_specs: upper bound on the number of implants inserted per call.
    :param severity: streak severity -- a scalar, or a ``(low, high)`` range sampled per call.
    :param intensity: implant HU value -- a scalar, or a ``(low, high)`` range sampled per implant.
    :param exclude_labels: labels that may not anchor an implant.
    :param include_labels: if given, the only labels that may anchor an implant.
    :param verbose: whether to warn when placement falls back to a random position.
    :param skip_minus_one_selected_label: skip implants that could not be anchored on a label.
    :param exclude_class: subset of ``("metal", "wire", "calcification")`` to exclude from the random choice.
    :param prob: probability of applying the transform.
    """

    def __init__(self, keys: KeysCollection, spacing, label_key: str = "label", max_n_specs: int = 2,
                severity: ValueOrRange = (0.1, 0.9), intensity: ValueOrRange = (1000, 5000),
                exclude_labels: Union[Sequence, int, None] = (0,),
                include_labels: Union[Sequence, int, None] = None, verbose: bool = False,
                skip_minus_one_selected_label: bool = True,
                exclude_class: Optional[Tuple[str, ...]] = None,
                prob: float = 1.0, allow_missing_keys: bool = False):
        core = RandomArtifactTransform(spacing=spacing, max_n_specs=max_n_specs, severity=severity,
                                       intensity=intensity,
                                       exclude_labels=exclude_labels, include_labels=include_labels,
                                       verbose=verbose,
                                       skip_minus_one_selected_label=skip_minus_one_selected_label,
                                       exclude_class=exclude_class)
        super().__init__(core, keys, label_key=label_key, prob=prob, allow_missing_keys=allow_missing_keys)
