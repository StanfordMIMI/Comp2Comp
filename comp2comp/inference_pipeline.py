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
    def __init__(self, config: Dict = None, inference_classes: List = None):
        self.config = config
        self.inference_classes = inference_classes

    def __call__(self):
        # print out the class names for each inference class
        print("")
        print("Inference pipeline:")
        for inference_class in self.inference_classes:
            print(inference_class.__repr__())
        print("")
        print("Running {} with input keys {}".format(self.inference_classes[0].__repr__(), 
            inspect.signature(self.inference_classes[0]).parameters.keys()))
        output = self.inference_classes[0]()
        print("Finished {} with output keys {}\n".format(self.inference_classes[0].__repr__(), 
            output.keys()))

        for inference_class in self.inference_classes[1:]:
            assert set(inspect.signature(inference_class).parameters.keys()) == set(output.keys()), \
                "Input to inference class, {}, does not have the correct parameters".format(inference_class.__repr__())

            print("Running {} with input keys {}".format(inference_class.__repr__(),
                inspect.signature(inference_class).parameters.keys()))
            output = inference_class(**output)
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

    dicom_loader = DicomLoader(args.dicom_dir)
    nifti_saver = NiftiSaver(args.output_dir)

    pipeline = InferencePipeline(inference_classes=[dicom_loader, nifti_saver])
    pipeline()

    print("Done.")

