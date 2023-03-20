# <img src="logo.png" width="40" height="40" /> Comp2Comp 
Comp2Comp is a library for extracting clinical insights from computed tomography scans. 

## Installation
```bash
# Install from local clone:
git clone https://github.com/StanfordMIMI/Comp2Comp/

# Install script requires Anaconda/Miniconda.
cd Comp2Comp && bin/install.sh
```
For installing on the Apple M1 chip, see [these instructions](https://github.com/StanfordMIMI/Comp2Comp/blob/master/Local%20Implementation%20%40%20M1%20arm64%20Silicon.md).

## Inference Pipeline
We have designed Comp2Comp to be highly extensible and to enable the development of complex clinically-relevant applications. We observed that many clinical applications require chaining several machine learning or other computational modules together to generate complex insights. The inference pipeline system is designed to make this easy. Furthermore, we seek to make the code readable and modular, so that the community can easily contribute to the project. 

The InferencePipeline class (comp2comp/inference_pipeline.py) is used to create inference pipelines, which are made up of a sequence of InferenceClass objects. When the InferencePipeline object is called, it sequentially calls the InferenceClasses that were provided to the constructor. 

The first argument of the __call__ function of InferenceClass must be the InferencePipeline object. This allows each InferenceClass object to access or set attributes of the InferencePipeline object that can be accessed by the subsequent InferenceClasses in the pipeline. Each InferenceClass object should return a dictionary where the keys of the dictionary should match any keyword arguments of the subsequent InferenceClass's __call__ function. If an InferenceClass object only sets attributes of the InferencePipeline object but does not return any value, an empty dictionary can be returned. 

Currently, Comp2Comp has two pipelines, muscle_adipose_tissue and spine_muscle_adipose_tissue, and two more pipelines, abdominal_aortic_aneurysm and contrast_phase_detection, are in progress.

## Current Pipelines
- muscle_adipose_tissuex
  - DICOM files within the input_path folder and subfolders of input_path will be processed.
- spine_muscle_adipose_tissue
  - input_path should contain a DICOM series or subfolders that contain DICOM series.

## In Progess
- abdominal_aortic_aneurysm
- contrast_phase_detection

## Basic Usage
```bash
bin/C2C <pipeline_name> --input_path <path/to/input/folder>
```

For running on slurm, modify the above commands as follow:
```bash
bin/C2C-slurm <pipeline_name> --input_path <path/to/input/folder>
```

## Inference Pipeline
We have designed Comp2Comp to be highly extensible and to enable the development of complex clinically-relevant applications. We observed that many clinical applications require chaining several machine learning or other computational modules together to generate complex insights. The inference pipeline system is designed to make this easy. Furthermore, we seek to make the code readable and modular, so that the community can easily contribute to the project. 

The InferencePipeline class (comp2comp/inference_pipeline.py) is used to create inference pipelines, which are made up of a sequence of InferenceClass objects. When the InferencePipeline object is called, it sequentially calls the InferenceClasses that were provided to the constructor. 

The first argument of the __call__ function of InferenceClass must be the InferencePipeline object. This allows each InferenceClass object to access or set attributes of the InferencePipeline object that can be accessed by the subsequent InferenceClasses in the pipeline. Each InferenceClass object should return a dictionary where the keys of the dictionary should match any keyword arguments of the subsequent InferenceClass's __call__ function. If an InferenceClass object only sets attributes of the InferencePipeline object but does not return any value, an empty dictionary can be returned. 

Currently, Comp2Comp has two pipelines, muscle_adipose_tissue and spine_muscle_adipose_tissue, and two more pipelines, abdominal_aortic_aneurysm and contrast_phase_detection, are in progress.

## Current Pipelines
- muscle_adipose_tissuex
  - DICOM files within the input_path folder and subfolders of input_path will be processed.
- spine_muscle_adipose_tissue
  - input_path should contain a DICOM series or subfolders that contain DICOM series.

## In Progess
- abdominal_aortic_aneurysm
- contrast_phase_detection

## Example image output from spine_muscle_adipose_tissue
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


