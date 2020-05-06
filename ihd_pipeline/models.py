import os
import enum
from typing import Dict, Optional, Sequence

import numpy as np
from keras.models import load_model

from ihd_pipeline.preferences import PREFERENCES


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $FVCORE_CACHE, if set
        2) otherwise ~/.torch/fvcore_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("IHD_PIPELINE_CACHE", "~/.ihd_pipeline/cache")
        )
    return cache_dir


class Models(enum.Enum):
    ABCT_V_0_0_1 = 1, "abct_v0.0.1", [""]
    STANFORD_V_0_0_1 = 2, "stanford_v0.0.1", [""]

    def __new__(
        cls,
        value: int,
        model_name: str,
        categories: Sequence[str],
        use_softmax: bool
    ):
        obj = object.__new__(cls)
        obj._value_ = value

        obj.model_name = model_name
        obj.categories = [x.lower() for x in categories]
        obj.use_softmax = use_softmax
        return obj

    def load_model(self):
        filename = os.path.join(
            PREFERENCES.MODEL_DIR, ".h5".format(self.model_name)
        )
        return load_model(filename)

    def preds_to_mask(self, preds):
        if self.use_softmax:
            # softmax
            labels = np.zeros_like(preds, dtype=np.uint8)
            l_argmax = np.argmax(preds, axis=-1)
            for c in range(labels.shape[-1]):
                labels[l_argmax == c, c] = 1
            return labels.astype(np.uint)
        else:
            # sigmoid
            return preds >= 0.5
