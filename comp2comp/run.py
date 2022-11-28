import logging
import os
import re
from typing import Dict, Sequence, Union

from comp2comp.metrics import CrossSectionalArea, HounsfieldUnits
from comp2comp.preferences import PREFERENCES

logger = logging.getLogger(__name__)


def format_output_path(
    file_path,
    save_dir: str = None,
    base_dirs: Sequence[str] = None,
    file_name: Sequence[str] = None,
):
    """Format output path for a given file.

    Args:
        file_path (str): File path.
        save_dir (str, optional): Save directory. Defaults to None.
        base_dirs (Sequence[str], optional): Base directories. Defaults to None.
        file_name (Sequence[str], optional): File name. Defaults to None.

    Returns:
        str: Output path.
    """
    if not save_dir:
        save_dir = PREFERENCES.OUTPUT_PATH

    dirname = os.path.dirname(file_path) if not save_dir else save_dir

    if save_dir and base_dirs:
        dirname: str = os.path.dirname(file_path)
        relative_dir = [
            dirname.split(bdir, 1)[1] for bdir in base_dirs if dirname.startswith(bdir)
        ][0]
        # Trim path separator from the path
        relative_dir = relative_dir.lstrip(os.path.sep)
        dirname = os.path.join(save_dir, relative_dir)

    if file_name is not None:
        return os.path.join(
            dirname,
            "{}.h5".format(file_name),
        )

    return os.path.join(
        dirname,
        "{}.h5".format(os.path.splitext(os.path.basename(file_path))[0]),
    )


# Function the returns a list of file names exluding
# the extention from the list of file paths
def get_file_names(files):
    """Get file names from a list of file paths.

    Args:
        files (list): List of file paths.

    Returns:
        list: List of file names.
    """
    file_names = []
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_names.append(file_name)
    return file_names


def find_files(
    root_dirs: Union[str, Sequence[str]],
    max_depth: int = None,
    exist_ok: bool = False,
    pattern: str = None,
):
    """Recursively search for files.

    To avoid recomputing experiments with results, set `exist_ok=False`.
    Results will be searched for in `PREFERENCES.OUTPUT_DIR` (if non-empty).

    Args:
        root_dirs (`str(s)`): Root folder(s) to search.
        max_depth (int, optional): Maximum depth to search.
        exist_ok (bool, optional): If `True`, recompute results for
            scans.
        pattern (str, optional): If specified, looks for files with names
            matching the pattern.

    Return:
        List[str]: Experiment directories to test.
    """

    def _get_files(depth: int, dir_name: str):
        if dir_name is None or not os.path.isdir(dir_name):
            return []

        if max_depth is not None and depth > max_depth:
            return []

        files = os.listdir(dir_name)
        ret_files = []
        for file in files:
            possible_dir = os.path.join(dir_name, file)
            if os.path.isdir(possible_dir):
                subfiles = _get_files(depth + 1, possible_dir)
                ret_files.extend(subfiles)
            elif os.path.isfile(possible_dir):
                if pattern and not re.match(pattern, possible_dir):
                    continue
                output_path = format_output_path(possible_dir)
                if not exist_ok and os.path.isfile(output_path):
                    logger.info(
                        "Skipping {} - results exist at {}".format(possible_dir, output_path)
                    )
                    continue
                ret_files.append(possible_dir)

        return ret_files

    out_files = []
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    for d in root_dirs:
        out_files.extend(_get_files(0, d))

    return sorted(set(out_files))


def compute_results(x, mask, categories: Dict, params: Dict):
    """Compute results for a given segmentation.

    Args:
        x (np.ndarray): Image.
        mask (np.ndarray): Segmentation mask.
        categories (Dict): Categories.
        params (Dict): Parameters.

    Returns:
        Dict: Results.
    """
    hu = HounsfieldUnits()
    spacing = params.get("spacing", None)
    csa_units = "mm^2" if spacing else ""
    csa = CrossSectionalArea(csa_units)

    hu_vals = hu(mask, x, category_dim=-1)
    csa_vals = csa(mask=mask, spacing=spacing, category_dim=-1)

    assert mask.shape[-1] == len(
        categories
    ), "{} categories found in mask, " "but only {} categories specified".format(
        mask.shape[-1], len(categories)
    )

    results = {
        cat: {
            "mask": mask[..., idx],
            hu.name(): hu_vals[idx],
            csa.name(): csa_vals[idx],
        }
        for idx, cat in enumerate(categories.keys())
    }

    return results


def get_dicom_paths_and_num(path):
    """Get all paths under a path that contain only dicom files.

    Args:
        path (str): Path to search.

    Returns:
        list: List of paths.
    """
    dicom_paths = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            if all([file.endswith(".dcm") for file in files]):
                dicom_paths.append((root, len(files)))
    return dicom_paths
