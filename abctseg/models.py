import numpy as np
from keras.models import load_model

import enum
import os
from abctseg.preferences import PREFERENCES
from typing import Sequence


class Models(enum.Enum):
    ABCT_V_0_0_1 = (
        1,
        "abCT_v0.0.1",
        ("muscle", "imat", "vat", "sat"),
        False,
        ("soft", "bone", "custom"),
    )
    STANFORD_V_0_0_1 = (
        2,
        "stanford_v0.0.1",
        ("background", "muscle", "bone", "vat", "sat"),
        True,
        ("soft", "bone", "custom"),
    )

    def __new__(
        cls,
        value: int,
        model_name: str,
        categories: Sequence[str],
        use_softmax: bool,
        windows: Sequence[str],
    ):
        obj = object.__new__(cls)
        obj._value_ = value

        obj.model_name = model_name
        obj.categories = [x.lower() for x in categories]
        obj.use_softmax = use_softmax
        obj.windows = windows
        return obj

    def load_model(self):
        filename = os.path.join(
            PREFERENCES.MODELS_DIR, "{}.h5".format(self.model_name)
        )
        return load_model(filename)

    def preds_to_mask(self, preds):
        if self.use_softmax:
            # softmax
            labels = np.zeros_like(preds, dtype=np.uint8)
            l_argmax = np.argmax(preds, axis=-1)
            for c in range(labels.shape[-1]):
                labels[l_argmax == c, c] = 1
            return labels.astype(np.bool)
        else:
            # sigmoid
            return preds >= 0.5
