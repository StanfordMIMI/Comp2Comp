import os

from yacs.config import CfgNode as CN

_PREFERENCES_FILE = os.path.join(
    os.path.join(os.path.dirname(__file__), "preferences.yaml")
)
_HOME_DIR = os.path.expanduser("~")


_C = CN()
_C.DATA_DIR = ""
_C.CACHE_DIR = os.path.join(_HOME_DIR, ".abctseg/cache")
_C.MODELS_DIR = ""  # TODO: UNDO THIS

_C.BATCH_SIZE = 16
_C.NUM_WORKERS = 0


def save_preferences(filename=None):
    if filename is None:
        filename = _PREFERENCES_FILE
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        f.write(PREFERENCES.dump())


PREFERENCES = _C
if not os.path.isfile(_PREFERENCES_FILE):
    save_preferences()
else:
    PREFERENCES.merge_from_file(_PREFERENCES_FILE)
