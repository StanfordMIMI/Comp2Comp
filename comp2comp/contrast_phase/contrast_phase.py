import os
import subprocess
import zipfile
from pathlib import Path
from time import time
from typing import Union

from totalsegmentator.libs import (  # download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.contrast_phase.contrast_inf import predict_phase
from comp2comp.inference_class_base import InferenceClass

# from totalsegmentatorv2.python_api import totalsegmentator



class ContrastPhaseDetection(InferenceClass):
    """Contrast Phase Detection."""

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def __call__(self, inference_pipeline):
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)
        self.model_dir = inference_pipeline.model_dir

        seg, img = self.run_segmentation(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            self.output_dir_segmentations + "s01.nii.gz",
            inference_pipeline.model_dir,
        )

        # segArray, imgArray = self.convertNibToNumpy(seg, img)

        imgNiftiPath = os.path.join(
            self.output_dir_segmentations, "converted_dcm.nii.gz"
        )
        segNiftPath = os.path.join(self.output_dir_segmentations, "s01.nii.gz")

        predict_phase(segNiftPath, imgNiftiPath, outputPath=self.output_dir)

        return {}

    def run_segmentation(
        self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir
    ):
        """Run segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [251]

        setup_nnunet()
        # for task_id in [251]:
        #     download_pretrained_weights(task_id)

        # download with weight for id 251
        self.download_pretrained_weights_updated(task_id[0])

        from totalsegmentator.nnunet import nnUNet_predict_image

        with nostdout():
            img, seg = nnUNet_predict_image(
                input_path,
                output_path,
                task_id,
                model=model,
                folds=folds,
                trainer=trainer,
                tta=False,
                multilabel_image=True,
                resample=1.5,
                crop=None,
                crop_path=crop_path,
                task_name="total",
                nora_tag=None,
                preview=False,
                nr_threads_resampling=1,
                nr_threads_saving=6,
                quiet=False,
                verbose=False,
                test=0,
            )

        #  seg = totalsegmentator(
        #     input = os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
        #     output = os.path.join(self.output_dir_segmentations, "segmentation.nii"),
        #     task_ids = [293],
        #     ml = True,
        #     nr_thr_resamp = 1,
        #     nr_thr_saving = 6,
        #     fast = False,
        #     nora_tag = "None",
        #     preview = False,
        #     task = "total",
        #     roi_subset = None,
        #     statistics = False,
        #     radiomics = False,
        #     crop_path = None,
        #     body_seg = False,
        #     force_split = False,
        #     output_type = "nifti",
        #     quiet = False,
        #     verbose = False,
        #     test = 0,
        #     skip_saving = True,
        #     device = "gpu",
        #     license_number = None,
        #     statistics_exclude_masks_at_border = True,
        #     no_derived_masks = False,
        #     v1_order = False,
        # )
        end = time()

        # Log total time for spine segmentation
        print(f"Total time for segmentation: {end-st:.2f}s.")

        # return seg, img
        return seg, img

    def download_pretrained_weights_updated(self, task_id):
        """
        Download the weights with curl to resolve problems
        with downloading from Zenodo
        """
        home_path = Path(os.environ["SCRATCH"])
        config_dir = home_path / ".totalsegmentator/nnunet/results/nnUNet"
        (config_dir / "3d_fullres").mkdir(exist_ok=True, parents=True)
        (config_dir / "2d").mkdir(exist_ok=True, parents=True)

        url = "https://zenodo.org/records/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
        config_dir = config_dir / "3d_fullres"
        weights_path = config_dir / "Task251_TotalSegmentator_part1_organs_1139subj"
        tempfile = config_dir / "tmp_download_file.zip"

        if not weights_path.exists():
            print("Downloading weights..")
            subprocess.run(["curl", "-L", url, "-o", tempfile], check=True)

            print("Unzipping..")
            with zipfile.ZipFile(config_dir / "tmp_download_file.zip", "r") as zip_f:
                zip_f.extractall(config_dir)
            # print(f"  downloaded in {time.time()-st:.2f}s")
            if tempfile.exists():
                os.remove(tempfile)
            print("Done.")
        else:
            print("Weights are already downloaded")

    def convertNibToNumpy(self, TSNib, ImageNib):
        """Convert nifti to numpy array.

        Args:
            TSNib (nibabel.nifti1.Nifti1Image): TotalSegmentator output.
            ImageNib (nibabel.nifti1.Nifti1Image): Input image.

        Returns:
            numpy.ndarray: TotalSegmentator output.
            numpy.ndarray: Input image.
        """
        TS_array = TSNib.get_fdata()
        img_array = ImageNib.get_fdata()
        return TS_array, img_array
