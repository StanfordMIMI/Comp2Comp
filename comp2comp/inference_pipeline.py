"""
@author: louisblankemeier
"""

import inspect
import os
from typing import Dict, List

import nibabel as nib

from comp2comp.inference_class_base import InferenceClass
from comp2comp.io.io import DicomLoader, NiftiSaver


class InferencePipeline(InferenceClass):
    """Inference pipeline."""

    def __init__(
        self, inference_classes: List = None, config: Dict = None, args: Dict = None
    ):
        self.config = config
        self.args = args
        # assign values from config to attributes
        if self.config is not None:
            for key, value in self.config.items():
                setattr(self, key, value)

        self.inference_classes = inference_classes

    def __call__(self, inference_pipeline=None, input_path: str = None, **kwargs):
        # print out the class names for each inference class
        print("")
        print("Inference pipeline:")
        for i, inference_class in enumerate(self.inference_classes):
            print(f"({i + 1}) {inference_class.__repr__()}")
        print("")

        print("Starting inference pipeline for:\n")

        # pick whether we’re writing onto self or the wrapped pipeline
        target = inference_pipeline or self

        # 1) stash input_path if provided
        if input_path is not None:
            target.input_path = input_path

        # 2) stash everything else (output_dir, model_dir, etc.)
        for key, value in kwargs.items():
            setattr(target, key, value)

        output = {}
        for inference_class in self.inference_classes:
            function_keys = set(inspect.signature(inference_class).parameters.keys())
            function_keys.remove("inference_pipeline")

            if "kwargs" in function_keys:
                function_keys.remove("kwargs")

            assert function_keys == set(
                output.keys()
            ), "Input to inference class, {}, does not have the correct parameters".format(
                inference_class.__repr__()
            )

            print(
                "Running {} with input keys {}".format(
                    inference_class.__repr__(),
                    inspect.signature(inference_class).parameters.keys(),
                )
            )

            if inference_pipeline:
                output = inference_class(
                    inference_pipeline=inference_pipeline, **output
                )
            else:
                output = inference_class(inference_pipeline=self, **output)

            # if not the last inference class, check that the output keys are correct
            if inference_class != self.inference_classes[-1]:
                print(
                    "Finished {} with output keys {}\n".format(
                        inference_class.__repr__(), output.keys()
                    )
                )

        print("Inference pipeline finished.\n")

        return output

    def saveArrToNifti(self, arr, path):
        """
        Saves an array to nifti using the CT as reference

        Args:
            arr (ndarray): input array.
            path (str, Path): full save path.

        Returns:
            None.

        """
        img = nib.Nifti1Image(
            arr, self.medical_volume.affine, self.medical_volume.header
        )
        nib.save(img, path)


if __name__ == "__main__":
    """Example usage of InferencePipeline."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file_path = os.path.join(output_dir, "test.nii.gz")

    pipeline = InferencePipeline(
        [DicomLoader(args.dicom_dir), NiftiSaver()],
        config={"output_dir": output_file_path},
    )
    pipeline()

    print("Done.")
