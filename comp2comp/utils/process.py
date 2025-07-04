"""
@author: louisblankemeier
"""

import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from comp2comp.io import io_utils


def find_common_root(paths):
    paths_with_sep = [path if path.endswith("/") else path + "/" for path in paths]

    # Find common prefix, ensuring it ends with a directory separator
    common_root = os.path.commonprefix(paths_with_sep)
    common_root
    if not common_root.endswith("/"):
        # Find the last separator to correctly identify the common root directory
        common_root = common_root[: common_root.rfind("/") + 1]

    return common_root


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

    path_and_num = io_utils.get_dicom_or_nifti_paths_and_num(args.input_path)

    # in case input is a .txt file we need to find the common root of the files
    if args.input_path.endswith(".txt"):
        all_paths = [p[0] for p in path_and_num]
        common_root = find_common_root(all_paths)

    for path, num in path_and_num:

        try:
            st = time.time()

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
                if args.input_path.endswith(".txt"):
                    output_dir = Path(
                        os.path.join(
                            output_path,
                            os.path.relpath(os.path.normpath(path), common_root),
                        )
                    )
                else:
                    output_dir = Path(
                        os.path.join(
                            output_path,
                            Path(os.path.basename(os.path.normpath(args.input_path))),
                            os.path.relpath(
                                os.path.normpath(path),
                                os.path.normpath(args.input_path),
                            ),
                        )
                    )

            if not os.path.exists(output_dir):
                output_dir.mkdir(parents=True)

            pipeline = pipeline_builder(path, args)

            pipeline(
                input_path=path,
                output_dir=output_dir,
                model_dir=model_dir
            )

            if not args.save_segmentations:
                # remove the segmentations folder
                segmentations_dir = os.path.join(output_dir, "segmentations")
                if os.path.exists(segmentations_dir):
                    shutil.rmtree(segmentations_dir)

            print(f"Finished processing {path} in {time.time() - st:.1f} seconds\n")
            print("Output was saved to:")
            print(output_dir)

        except Exception:
            print(f"ERROR PROCESSING {path}\n")
            traceback.print_exc()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            # remove parent folder if empty
            if len(os.listdir(os.path.dirname(output_dir))) == 0:
                shutil.rmtree(os.path.dirname(output_dir))
            continue
