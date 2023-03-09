from typing import List, Dict, Tuple, Union, Optional, Any
import inspect
import logging
import os
import sys
from pathlib import Path
import dosma as dm

from comp2comp.io import DicomLoader, NiftiSaver

class InferencePipeline:
    """Inference pipeline.
    """
    def __init__(self, inference_classes: List = None, config: Dict = None):
        self.config = config
        # assign values from config to attributes
        if self.config is not None:
            for key, value in self.config.items():
                setattr(self, key, value)
                
        self.inference_classes = inference_classes

    def __call__(self, **kwargs):
        # print out the class names for each inference class
        print("Inference pipeline:")
        for inference_class in self.inference_classes:
            print(inference_class.__repr__())
        print("")

        output = kwargs
        for inference_class in self.inference_classes:
            assert set(inspect.signature(inference_class).parameters.keys()) == set(output.keys()), \
                "Input to inference class, {}, does not have the correct parameters".format(inference_class.__repr__())

            print("Running {} with input keys {}".format(inference_class.__repr__(),
                inspect.signature(inference_class).parameters.keys()))
            output = inference_class(self, **output)
            print("Finished {} with output keys {}\n".format(inference_class.__repr__(), 
                output.keys()))

        return output


if __name__ == "__main__":
    """Example usage of InferencePipeline.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)

    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if not os.path.exists(os.path.join(default_output_dir, "outputs")):
        os.mkdir(os.path.join(default_output_dir, "outputs"))
    default_output_dir = os.path.join(default_output_dir, "outputs/test.nii.gz")

    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    args = parser.parse_args()

    pipeline = InferencePipeline([
                                DicomLoader(args.dicom_dir),
                                NiftiSaver(args.output_dir)
                                ])
    pipeline()

    print("Done.")

