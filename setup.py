#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from os import path

from setuptools import find_packages, setup


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "comp2comp", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][
        0
    ]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("ABCTSEG_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [line for line in init_py if not line.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="comp2comp",
    version=get_version(),
    author="StanfordMIMI",
    url="https://github.com/StanfordMIMI/Comp2Comp",
    description="Computed tomography to body composition.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.9",
    install_requires=[
        "pydicom",
        "moviepy",
        "numpy==1.23.5",
        "h5py",
        "tabulate",
        "tqdm",
        "silx",
        "yacs",
        "pandas",
        "dosma",
        "opencv-python",
        "huggingface_hub",
        "pycocotools",
        "wget",
        "tensorflow==2.12.0",
        "totalsegmentator @ git+https://github.com/StanfordMIMI/TotalSegmentator.git",
        "totalsegmentatorv2 @ git+https://github.com/StanfordMIMI/TotalSegmentatorV2.git",
        "blosc2",
        "acvl-utils==0.2",
        "dicom2nifti<2.6",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "xgboost",
    ],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": [
            # Formatting
            "flake8",
            "isort",
            "black==22.8.0",
            "flake8-bugbear",
            "flake8-comprehensions",
            # Docs
            "mock",
            "sphinx",
            "sphinx-rtd-theme",
            "recommonmark",
            "myst-parser",
        ],
        "contrast_phase": ["xgboost"],
    },
)
