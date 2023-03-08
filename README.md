# <img src="logo.png" width="40" height="40" /> Comp2Comp 
Comp2Comp is a library for extracting body composition measures from computed tomography scans. 

## Installation
```bash
# Install from local clone:
git clone https://github.com/StanfordMIMI/Comp2Comp/

# Install script requires Anaconda/Miniconda.
cd Comp2Comp && bin/install.sh
```
For installing on the Apple M1 chip, see [these instructions](https://github.com/StanfordMIMI/Comp2Comp/blob/master/Local%20Implementation%20%40%20M1%20arm64%20Silicon.md).

## Basic Usage
```bash
# To run 3D body composition, use the following command. INPUT_PATH should contain a DICOM series or subfolders that contain DICOM series.
bin/C2C process_3d INPUT_PATH path/to/input/folder

# To run 2D body composition, using the following command. DICOM files within the INPUT_PATH folder and subfolders of INPUT_PATH will be processed.
bin/C2C process_2d INPUT_PATH path/to/input/folder
```

For running on slurm, modify the above commands as follow:
```bash
# For 3D processing with slurm, can use the following:
bin/C2C-slurm process_3d INPUT_PATH path/to/input/folder

# For 2D processing with slurm, can use the following:
bin/C2C-slurm process_2d INPUT_PATH path/to/input/folder
```

## Example Image Output
![Alt text](figures/panel_example.png?raw=true "Comp2Comp Panel Example")

## Citation
``` 
@article{blankemeier2023comp2comp,
  title={Comp2Comp: Open-Source Body Composition Assessment on Computed Tomography},
  author={Blankemeier, Louis and Desai, Arjun and Chaves, Juan Manuel Zambrano and Wentland, Andrew and Yao, Sally and Reis, Eduardo and Jensen, Malte and Bahl, Bhanushree and Arora, Khushboo and Patel, Bhavik N and others},
  journal={arXiv preprint arXiv:2302.06568},
  year={2023}
}
```

In addition to Comp2Comp, please consider citing TotalSegmentator:
```
@article{wasserthal2022totalsegmentator,
  title={TotalSegmentator: robust segmentation of 104 anatomical structures in CT images},
  author={Wasserthal, Jakob and Meyer, Manfred and Breit, Hanns-Christian and Cyriac, Joshy and Yang, Shan and Segeroth, Martin},
  journal={arXiv preprint arXiv:2208.05868},
  year={2022}
}
```


