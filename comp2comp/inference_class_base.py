from Typing import Dict


class InferenceClass:
    """Base class for inference classes."""

    def __init__(self):
        pass

    def __call__(self) -> Dict:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__
