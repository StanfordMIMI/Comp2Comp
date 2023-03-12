from typing import List, Dict, Tuple, Union, Optional, Any
import inspect
import logging
import os
import sys
from pathlib import Path


class InferenceClass():
    """Base class for inference classes.
    """
    def __init__(self):
        pass

    def __call__(self) -> Dict:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__
