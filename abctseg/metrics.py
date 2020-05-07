from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union

import numpy as np


def flatten_non_category_dims(
    xs: Union[np.ndarray, Sequence[np.ndarray]], category_dim: int = None
):
    """Flattens all non-category dimensions into a single dimension.

    Args:
        xs (ndarrays): Sequence of ndarrays with the same category dimension.
        category_dim: The dimension/axis corresponding to different categories.
            i.e. `C`. If `None`, behaves like `np.flatten(x)`.

    Returns:
        ndarray: Shape (C, -1) if `category_dim` specified else shape (-1,)
    """
    single_item = isinstance(xs, np.ndarray)
    if single_item:
        xs = [xs]

    if category_dim is not None:
        dims = (xs[0].shape[category_dim], -1)
        xs = (np.moveaxis(x, category_dim, 0).reshape(dims) for x in xs)
    else:
        xs = (x.flatten() for x in xs)

    if single_item:
        return list(xs)[0]
    else:
        return xs


class Metric(Callable, ABC):
    """Interface for new metrics.

    A metric should be implemented as a callable with explicitly defined
    arguments. In other words, metrics should not have `**kwargs` or `**args`
    options in the `__call__` method.

    While not explicitly constrained to the return type, metrics typically
    return float value(s). The number of values returned corresponds to the
    number of categories.

    * metrics should have different name() for different functionality.
    * `category_dim` duck type if metric can process multiple categories at
        once.

    To compute metrics:

    .. code-block:: python

        metric = Metric()
        results = metric(...)
    """

    def __init__(self, units: str = ""):
        self.units = units

    def name(self):
        return type(self).__name__

    def display_name(self):
        """Name to use for pretty printing and display purposes.
        """
        name = self.name()
        return "{} {}".format(name, self.units) if self.units else name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class HounsfieldUnits(Metric):
    FULL_NAME = "Hounsfield Unit"

    def __init__(self, units="hu"):
        super().__init__(units)

    def __call__(self, mask, x, category_dim: int = None):
        mask = mask.astype(np.bool)
        if category_dim is None:
            return np.mean(x[mask])

        assert category_dim == -1
        num_classes = mask.shape[-1]

        return np.array([np.mean(x[mask[..., c]]) for c in range(num_classes)])

    def name(self):
        return self.FULL_NAME


class CrossSectionalArea(Metric):
    def __call__(self, mask, spacing=None, category_dim: int = None):
        pixel_area = np.prod(spacing) if spacing else 1
        mask = mask.astype(np.bool)
        mask = flatten_non_category_dims(mask, category_dim)

        return pixel_area * np.count_nonzero(mask, -1)

    def name(self):
        if self.units:
            return "Cross-sectional Area ({})".format(self.units)
        else:
            return "Cross-sectional Area"
