import enum
import os
from typing import Dict, Sequence

import numpy as np
from huggingface_hub import hf_hub_download
from keras.models import load_model
import wget
from pathlib import Path

from comp2comp.preferences import PREFERENCES


class Models(enum.Enum):
    """
    ABCT_V_0_0_1 = (
        1,
        "abCT_v0.0.1",
        ("muscle", "imat", "vat", "sat"),
        False,
        ("soft", "bone", "custom"),
    )
    """

    STANFORD_V_0_0_1 = (
        2,
        "stanford_v0.0.1",
        # ("background", "muscle", "bone", "vat", "sat", "imat"),
        # Category name mapped to channel index
        {"muscle": 1, "vat": 3, "sat": 4, "imat": 5},
        True,
        ("soft", "bone", "custom"),
    )
    TS_SPINE = (
        3,
        "ts_spine",
        # Category name mapped to channel index
        {"L5": 18, "L4": 19, "L3": 20, "L2": 21, "L1": 22, "T12": 23},
        False,
        (),
    )

    def __new__(
        cls,
        value: int,
        model_name: str,
        categories: Dict[str, int],
        use_softmax: bool,
        windows: Sequence[str],
    ):
        obj = object.__new__(cls)
        obj._value_ = value

        obj.model_name = model_name
        obj.categories = categories
        obj.use_softmax = use_softmax
        obj.windows = windows
        return obj

    def find_model_weights():
        """Find the model weights in the models directory.

        Returns:
            str: Path to model weights.
        """
        for root, dirs, files in os.walk(PREFERENCES.MODELS_DIR):
            for file in files:
                if file.endswith(".h5"):
                    filename = os.path.join(root, file)
        return filename

    def load_model(self, logger):
        """Load the model from the models directory.

        Args:
            logger (logging.Logger): Logger.

        Returns:
            keras.models.Model: Model.
        """
        # hf_hub_download(
        #     repo_id="lblankem/stanford_abct_v0.0.1",
        #     filename="stanford_v0.0.1.h5",
        #     cache_dir=PREFERENCES.MODELS_DIR,
             # use_auth_token=PREFERENCES.HF_TOKEN,
        # )
        
        logger.info("Downloading muscle/fat model from hugging face")
        Path(PREFERENCES.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        weights_file_name = wget.download("https://huggingface.co/stanfordmimi/stanford_abct_v0.0.1/resolve/main/stanford_v0.0.1.h5", out=os.path.join(PREFERENCES.MODELS_DIR, "stanford_v0.0.1.h5"))
        filename = Models.find_model_weights()
        
        logger.info("Loading muscle/fat model from {}".format(filename))
        return load_model(filename)

    def preds_to_mask(self, preds):
        """Convert model predictions to a mask.

        Args:
            preds (np.ndarray): Model predictions.

        Returns:
            np.ndarray: Mask.
        """
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

    @staticmethod
    def model_from_name(model_name):
        """Get the model enum from the model name.

        Args:
            model_name (str): Model name.

        Returns:
            Models: Model enum.
        """
        for model in Models:
            if model.model_name == model_name:
                return model
        return None

    @staticmethod
    def find_model_weights():
        for root, _, files in os.walk(PREFERENCES.MODELS_DIR):
            for file in files:
                if file.endswith(".h5"):
                    filename = os.path.join(root, file)
        return filename
