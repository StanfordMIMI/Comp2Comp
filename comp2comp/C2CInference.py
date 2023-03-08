from typing import List, Dict, Tuple, Union, Optional, Any

class InferencePipeline:
    """Inference pipeline.
    """
    def __init__(self, config: Dict, inference_classes: List[InferenceClass]):
        self.config = config
        self.inference_classes = inference_classes

    def __call__(self):
        assert len(inspect.signature(self.inference_classes[0]).parameters) == 0
        output = inference_class[0]()
        for inference_class in self.inference_classes[1:]:
            output = inference_class(output)
        return output


class InferenceClass():
    """Base class for inference classes.
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

