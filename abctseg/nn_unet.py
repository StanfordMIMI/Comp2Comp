import os
import sys
import random
import string
import time
import shutil
import subprocess
from pathlib import Path
from os.path import join
import numpy as np
import nibabel as nib
from time import time
import dosma as dm

from totalsegmentator.libs import setup_nnunet, download_pretrained_weights
from totalsegmentator.libs import nostdout
from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences

def spine_seg():
    st = time()
    os.environ["SCRATCH"] = PREFERENCES.CACHE_DIR
    input_dir = PREFERENCES.INPUT_DIR

    segmentations_path = Path(PREFERENCES.OUTPUT_DIR) / "segmentations"
    segmentations_path.mkdir(exist_ok=True)
    output_dir = str(segmentations_path / "spine.nii.gz")

    task_id = [252]
    model = "3d_fullres"
    folds = [0]
    trainer = "nnUNetTrainerV2_ep4000_nomirror"
    crop_path = None

    setup_nnunet()
    download_pretrained_weights(task_id[0])

    from totalsegmentator.nnunet import nnUNet_predict_image

    with nostdout():

        seg, mvs = nnUNet_predict_image(input_dir, output_dir, task_id, model=model, folds=folds,
                                trainer=trainer, tta=False, multilabel_image=True, resample=1.5,
                                crop=None, crop_path=crop_path, task_name="total", nora_tag=None, preview=False, 
                                nr_threads_resampling=1, nr_threads_saving=6, 
                                quiet=False, verbose=False, test=0)
    end = time()
    print(f"\nTotal time for spine segmentation: {end-st:.2f}s.\n")

    return seg, mvs