# Comp2Comp
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
# To run 3D body composition:
bin/C2C process_3d --INPUT_PATH path/to/input/folder
# To run 2D body composition:
bin/C2C process_2d --INPUT_PATH path/to/input/folder
# To submit scripts to slurm, can use the following for 3D processing:
bin/C2C-slurm process_3d --INPUT_PATH path/to/input/folder
# For 2D processing with slurm, can use the following:
bin/C2C-slurm process_2d --INPUT_PATH path/to/input/folder
```
