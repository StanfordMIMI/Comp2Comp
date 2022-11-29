# <img src="logo.png" width="40" height="40" /> Comp2Comp 
Comp2Comp is a library for extracting body composition measures from computed tomography scans. 

## Installation
```bash
# Install from local clone:
git clone -b rsna_mvp https://github.com/StanfordMIMI/Comp2Comp/
# Install script requires Anaconda/Miniconda.
cd Comp2Comp && bin/install.sh
```

## Basic Usage
```bash
# To run 3D body composition, use the following command. INPUT_PATH should contain a DICOM series or subfolders that contain DICOM series.
bin/C2C process_3d INPUT_PATH path/to/input/folder
# To run 2D body composition, using the following command. DICOM files within the INPUT_PATH folder and subfolders of INPUT_PATH will be processed.
bin/C2C process_2d INPUT_PATH path/to/input/folder

# For 3D processing with slurm, can use the following:
bin/C2C-slurm process_3d INPUT_PATH path/to/input/folder
# For 2D processing with slurm, can use the following:
bin/C2C-slurm process_2d INPUT_PATH path/to/input/folder
```
