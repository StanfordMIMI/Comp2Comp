import logging
import os
from pathlib import Path
from time import time
from typing import Union

from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

def spine_seg(
    logger: logging.Logger,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
):
    """Run spine segmentation.

    Args:
        logger (logging.Logger): Logger.
        input_path (Union[str, Path]): Input path.
        output_path (Union[str, Path]): Output path.
    """

    logger.info("Segmenting spine...")
    st = time()
    os.environ["SCRATCH"] = PREFERENCES.CACHE_DIR

    # Setup nnunet
    task_id = [252]
    model = "3d_fullres"
    folds = [0]
    trainer = "nnUNetTrainerV2_ep4000_nomirror"
    crop_path = None

    setup_nnunet()
    download_pretrained_weights(task_id[0])

    from totalsegmentator.nnunet import nnUNet_predict_image

    with nostdout():

        seg, mvs = nnUNet_predict_image(
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
    end = time()

    # Log total time for spine segmentation
    logger.info(f"Total time for spine segmentation: {end-st:.2f}s.")

    return seg, mvs
