
from typing import Any, Dict, Optional

from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform

from ctaug._base import DictTransform


class CTAugDictAdapter(MapTransform, RandomizableTransform):
    """Wraps a CTAug :class:`~ctaug._base.DictTransform` as a MONAI dictionary transform.

    ``keys`` selects which entries of the MONAI data dict are treated as the image to augment
    (each is run through the wrapped transform independently). If ``label_key`` is given, that
    entry is passed to the wrapped transform as ``segmentation`` (used to anchor artifact
    placement, or to be shifted alongside the image for motion transforms) and written back.
    """

    backend = []

    def __init__(self, core: DictTransform, keys: KeysCollection, label_key: Optional[str] = None,
                require_label: bool = False, prob: float = 1.0, allow_missing_keys: bool = False,
                info_key_suffix: str = "_ctaug_info"):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self._core = core
        self.label_key = label_key
        self.require_label = require_label
        self.info_key_suffix = info_key_suffix

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.key_iterator(d):
            core_input = {"image": d[key]}
            if self.label_key is not None:
                if self.label_key in d:
                    core_input["segmentation"] = d[self.label_key]
                elif self.require_label:
                    raise KeyError(
                        f"'{self.label_key}' not found in data but is required by "
                        f"{type(self._core).__name__}"
                    )

            out = self._core(**core_input)

            d[key] = out["image"]
            if self.label_key is not None and "segmentation" in out:
                d[self.label_key] = out["segmentation"]

            info = out.get("info")
            if info is None:
                info = next((v for k, v in out.items() if k.endswith("_info")), None)
            d[f"{key}{self.info_key_suffix}"] = info

        return d
