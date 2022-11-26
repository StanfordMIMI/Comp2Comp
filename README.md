# Comp2Comp ![Alt text](logo.png = 10x10)
Comp2Comp is a library for extracting body composition measures from computed tomography scans. 

## Installation
```bash
# Install it from a local clone:
git clone -b rsna_mvp https://github.com/StanfordMIMI/abCTSeg/
cd abCTSeg 
# Install script requires Anaconda/Miniconda.
bin/install.sh
```

## Basic Usage
```bash
# To run 3D body composition, use the following command. For 3D processing, INPUT_PATH should be a path to a folder that contains a series of DICOM files or subfolders that contain DICOM series.
bin/C2C process_3d INPUT_PATH path/to/input/folder
# To run 2D body composition, using the following command. For 2D processing, DICOM files within the INPUT_PATH folder and subfolders of INPUT_PATH will be processed.
bin/C2C process_2d INPUT_PATH path/to/input/folder
# To submit scripts to slurm, can use the following for 3D processing:
bin/C2C-slurm process_3d INPUT_PATH path/to/input/folder
# For 2D processing with slurm, can use the following:
bin/C2C-slurm process_2d INPUT_PATH path/to/input/folder
```
