# <img src="logo.png" width="40" height="40" /> Comp2Comp 
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/StanfordMIMI/Comp2Comp/format.yml?branch=master)
[![Documentation Status](https://readthedocs.org/projects/comp2comp/badge/?version=latest)](https://comp2comp.readthedocs.io/en/latest/?badge=latest)

[**Paper**](https://arxiv.org/abs/2302.06568)
| [**Installation**](#installation)
| [**Basic Usage**](#basic_usage)
| [**Inference Pipelines**](#basic_usage)
| [**Contribute**](#contribute)
| [**Citation**](#citation)

Comp2Comp is a library for extracting clinical insights from computed tomography scans.

## Installation
<a name="installation"></a>
```bash
git clone https://github.com/StanfordMIMI/Comp2Comp/

# Install script requires Anaconda/Miniconda.
cd Comp2Comp && bin/install.sh
```

Alternatively, Comp2Comp can be installed with `pip`:
```bash
git clone https://github.com/StanfordMIMI/Comp2Comp/
cd Comp2Comp
conda create -n c2c_env python=3.9
conda activate c2c_env
pip install -e .
```

For installing on the Apple M1 chip, see [these instructions](https://github.com/StanfordMIMI/Comp2Comp/blob/master/docs/Local%20Implementation%20%40%20M1%20arm64%20Silicon.md).

## Basic Usage
<a name="basic_usage"></a>
```bash
bin/C2C <pipeline_name> -i <path/to/input/folder>
```

For running on slurm, modify the above commands as follow:
```bash
bin/C2C-slurm <pipeline_name> -i <path/to/input/folder>
```

## Inference Pipelines
<a name="inference_pipeline"></a>
We have designed Comp2Comp to be highly extensible and to enable the development of complex clinically-relevant applications. We observed that many clinical applications require chaining several machine learning or other computational modules together to generate complex insights. The inference pipeline system is designed to make this easy. Furthermore, we seek to make the code readable and modular, so that the community can easily contribute to the project. 

The [`InferencePipeline` class](comp2comp/inference_pipeline.py) is used to create inference pipelines, which are made up of a sequence of [`InferenceClass` objects](comp2comp/inference_class_base.py). When the `InferencePipeline` object is called, it sequentially calls the `InferenceClasses` that were provided to the constructor. 

The first argument of the `__call__` function of `InferenceClass` must be the `InferencePipeline` object. This allows each `InferenceClass` object to access or set attributes of the `InferencePipeline` object that can be accessed by the subsequent `InferenceClass` objects in the pipeline. Each `InferenceClass` object should return a dictionary where the keys of the dictionary should match the keyword arguments of the subsequent `InferenceClass's` `__call__` function. If an `InferenceClass` object only sets attributes of the `InferencePipeline` object but does not return any value, an empty dictionary can be returned. 

Below are the inference pipelines currently supported by Comp2Comp.

## End-to-End Spine, Muscle, and Adipose Tissue Analysis at T12-L5

### Usage
```bash
bin/C2C spine_muscle_adipose_tissue -i <path/to/input/folder>
```
- input_path should contain a DICOM series or subfolders that contain DICOM series.

### Example Output Image
<p align="center">
  <img src="figures/spine_muscle_adipose_tissue_example.png" height="300">
</p>

## Spine Bone Mineral Density from 3D Trabecular Bone Regions at T12-L5

### Usage
```bash
bin/C2C spine -i <path/to/input/folder>
```
- input_path should contain a DICOM series or subfolders that contain DICOM series.

### Example Output Image
<p align="center">
  <img src="figures/spine_example.png" height="300">
</p>

## Abdominal Aortic Calcification Segmentation

### Usage
```bash
bin/C2C aortic_calcium -i <path/to/input/folder> -o <path/to/input/folder> --threshold --mosaic-type
```
The input path should contain a DICOM series or subfolders that contain DICOM series or a nifty file.
- The threshold can be controlled with `--threshold` and be either an integer HU threshold, "adataptive" or "agatson".
  - If "agatson" is used, agatson score is calculated and a threshold of 130 HU is used 
- Aortic calcifications are divided into abdominal and thoracic at the end of the T12 level
- Segmentation masks for the aortic calcium, the dilated aorta mask, and the T12 seperation plane are saved in ./segmentation_masks/
- Metrics on an aggregated and individual level for the calcifications are written to .csv files in ./metrics/
- Visualizations are saved to ./images/
  - The visualization presents coronal and sagittal MIP projections with the aorta overlay, featuring a heat map of calcifications alongside extracted calcification metrics. Below is a mosaic of each aortic slice with calcifications.  
  - The mosaic will default show all slices with califications but a subset at each vertebra level can be used instead with `--mosaic-type vertebrae` 

<p align="center">
  <img src="figures/aortic_calcium_overview.png" height="500">
</p>

### Example Output
```
Statistics on aortic calcifications:
Abdominal:
Total number:            21
Total volume (cm³):      1.042
Mean HU:                 218.6+/-91.4
Median HU:               195.6+/-65.8
Max HU:                  449.4+/-368.6
Mean volume (cm³):       0.050+/-0.100
Median volume (cm³):     0.006
Max volume (cm³):        0.425
Min volume (cm³):        0.002
Threshold (HU):          130.000
% Calcified aorta        3.429
Agatston score:          4224.7


Thoracic:
Total number:            5
Total volume (cm³):      0.012
Mean HU:                 171.6+/-41.0
Median HU:               168.5+/-42.7
Max HU:                  215.8+/-87.1
Mean volume (cm³):       0.002+/-0.001
Median volume (cm³):     0.002
Max volume (cm³):        0.004
Min volume (cm³):        0.002
Threshold (HU):          130.000
% Calcified aorta        0.026
Agatston score:          21.1
```

## AAA Segmentation and Maximum Diameter Measurement

### Usage
```bash
bin/C2C aaa -i <path/to/input/folder>
```
- input_path should contain a DICOM series or subfolders that contain DICOM series.

### Example Output Image (slice with largest diameter)
<p align="center">
  <img src="figures/aortic_aneurysm_example.png" height="300">
</p>

<div align="center">

| Example Output Video      | Example Output Graph     |
|-----------------------------|----------------------------|
| <p align="center"><img src="figures/aaa_segmentation_video.gif" height="300"></p> | <p align="center"><img src="figures/aaa_diameter_graph.png" height="300"></p> |

</div>

## Contrast Phase Detection

### Usage
```bash
bin/C2C contrast_phase -i <path/to/input/folder>
```
- input_path should contain a DICOM series or subfolders that contain DICOM series.
- This package has extra dependencies. To install those, run:
```bash
cd Comp2Comp
pip install -e '.[contrast_phase]'
```

## 3D Analysis of Liver, Spleen, and Pancreas

### Usage
```bash
bin/C2C liver_spleen_pancreas -i <path/to/input/folder>
```
- input_path should contain a DICOM series or subfolders that contain DICOM series.

### Example Output Image
<p align="center">
  <img src="figures/liver_spleen_pancreas_example.png" height="300">
</p>


## Contribute
<a name="contribute"></a>
We welcome all pull requests. If you have any issues, suggestions, or feedback, please open a new issue.

## Citation
<a name="citation"></a>
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


