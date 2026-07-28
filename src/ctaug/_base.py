from typing import Any, Dict


class DictTransform:
    """Base class for CTAug transforms operating on a dict of tensors.

    Subclasses implement ``__call__(self, **data_dict)`` and return the
    (possibly modified) ``data_dict``. This mirrors the dict-transform
    convention used by batchgenerators/MONAI pipelines without requiring
    either as a dependency, so CTAug's core stays torch-only.
    """

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        raise NotImplementedError
