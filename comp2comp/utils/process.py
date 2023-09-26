"""
@author: louisblankemeier
"""

import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import time

from comp2comp.io import io_utils


def process_2d(args, pipeline_builder):
    output_dir = Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../outputs",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    )
    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    pipeline = pipeline_builder(args)

    pipeline(output_dir=output_dir, model_dir=model_dir)


def process_3d(args, pipeline_builder):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../outputs"
        )

    if not args.overwrite_outputs:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_path, date_time)

    for path, num in io_utils.get_dicom_or_nifti_paths_and_num(args.input_path):
        try:
            st = time()

            if path.endswith(".nii") or path.endswith(".nii.gz"):
                print("Processing: ", path)

            else:
                print("Processing: ", path, " with ", num, " slices")
                min_slices = 30
                if num < min_slices:
                    print(f"Number of slices is less than {min_slices}, skipping\n")
                    continue

            print("")

            try:
                sys.stdout.flush()
            except Exception:
                pass

            if path.endswith(".nii") or path.endswith(".nii.gz"):
                folder_name = Path(os.path.basename(os.path.normpath(path)))
                # remove .nii or .nii.gz
                folder_name = os.path.normpath(
                    Path(str(folder_name).replace(".gz", "").replace(".nii", ""))
                )
                output_dir = Path(
                    os.path.join(
                        output_path,
                        folder_name,
                    )
                )

            else:
                output_dir = Path(
                    os.path.join(
                        output_path,
                        Path(os.path.basename(os.path.normpath(args.input_path))),
                        os.path.relpath(
                            os.path.normpath(path), os.path.normpath(args.input_path)
                        ),
                    )
                )

            if not os.path.exists(output_dir):
                output_dir.mkdir(parents=True)

            pipeline = pipeline_builder(path, args)

            pipeline(output_dir=output_dir, model_dir=model_dir)

            if not args.save_segmentations:
                # remove the segmentations folder
                segmentations_dir = os.path.join(output_dir, "segmentations")
                if os.path.exists(segmentations_dir):
                    shutil.rmtree(segmentations_dir)

            print(f"Finished processing {path} in {time() - st:.1f} seconds\n")

        except Exception:
            print(f"ERROR PROCESSING {path}\n")
            traceback.print_exc()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            continue
