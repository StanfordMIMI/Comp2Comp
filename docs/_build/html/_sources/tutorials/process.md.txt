# Processing scans
abCTSeg is built to simplify two critical tasks: 1. segmenting abdominal CT scans and 2. 
extracting key metrics that may be useful in downstream analysis. Below we discuss these
two tasks in more detail.

### Segmentation Models
There are many pre-trained networks for automatic segmentation of abdominal CT scans.
In abCTSeg, we support two networks that have shown high fidelity in both
segmentation and clinical metrics.

1. `stanford_v0.0.1`: This model segments muscle, bone, visceral fat (VAT), 
and subcutaneous fat (SAT).
2. `abCT_v0.0.1`: This model segments muscle, intramuscular fat (IMAT), VAT, and SAT.

Details on downloading and setting up these models are provided in 
[Getting Started with abCTSeg](getting_started.html).


### Calculating metrics
Two metrics that are popular in downstream CT analysis are hounsfield units (HU) and cross-sectional
area (CSA). Both metrics are implemented and automatically calculated for all segmented tissues when using
the command-line.


### Processing scans via the Command Line
abCTSeg is built to allow for easy access through the command line.
We provide a script in "abctseg/cli.py", which is the entry point for all command-line operations.
In this script, the `process` sub-command will segment scans in the appropriate DICOM format,
compute metrics, and save the data in a structured way. Below we explain the command-line arguments
in detail and provide some basic usage.

##### Command Line Arguments for `process`
For a summary of command-line arguments, run `python abctseg/cli.py process --help`. 
Please defer to the list below for more details.

- `--dicoms [str ...]`: An arbitrary number of dicoms and/or directories to search for dicoms.
- `--pattern [str]`: Regex pattern for selecting files if `--dicoms` specifies a directory.
- `--max-depth [int]`: Maximum depth to search directory if `--dicoms` specifies a directory.
- `--overwrite`: If flag is present, will overwrite results for files with the same name.
- `--num-gpus [int]`: Number of gpus to use. To use cpu, specify `0`.
- `--models [str ...]`: Names of models to use for segmentation. If multiple models are specified,
segmentation masks and metrics will be calculated using each model independently. i.e. no
ensembling is done.
- `--pp`: Use built-in post-processing. See section below for details

##### Understanding post-processing
The post-processing step is motivated by the underlying chemistry and spatial proximity of different tissues.
In particular, the chemical composition of muscle and intramuscular fat (IMAT) is quite different. However,
due to the sparse nature of IMAT, it can be difficult to segment between the two tissues. 

The post-processing step leverages the chemical difference between muscle and IMAT by the following filtration
algorithm. If a pixel is labeled as muscle but has a hounsfield denisty < -30, it is relabeled as IMAT. This may
improve classification at boundary regions.

From anecdotal experience, this post-processing step did not make a significant difference in accuracy among
average Dice, hounsfield density, or cross-sectional area.

##### Examples
Run segmentation on dicom files or directories with dicom files:

```bash
# Segment multiple dicom files.
python abctseg/cli.py \
    --dicoms /path/to/scan1/file /path/to/scan2/file \
    --models stanford_v0.0.1

# Segment all files in dir1 and dir2 ending with .dcm
python abctseg/cli.py \
    --dicoms /path/to/dir1 /path/to/dir2 --pattern ".*\.dcm" \
    --models stanford_v0.0.1
```

Use multiple models:

```bash
python abctseg/cli.py \
    --dicoms /path/to/scan1/file
    --models stanford_v0.0.1 abCT_v0.0.1
```

Change preferences (e.g. `BATCH_SIZE` and `OUTPUT_DIR`) for this run only:

```bash
python abctseg/cli.py \
    --dicoms /path/to/scan1/file
    --models stanford_v0.0.1 abCT_v0.0.1 \
    BATCH_SIZE 32 OUTPUT_DIR /path/to/dir
```

### Parsing results
After segmentation and metric calculation, the data is stored in the `h5` format per scan.
The h5 format leverages a dictionary-like hierarchy to store data. abCTSeg stores data in
the following top-down hierarchy - model -> category -> relevant data. A sample hierarchy is shown
below. The three data keys are `"mask"`, `"Cross-sectional Area (mm^2)"`, `"Hounsfield Unit"`. 

```
<MODEL1_NAME>
    | <CATEGORY1>
        | "mask": binary mask for this category
        | "Cross-sectional Area (mm^2)": the cross sectional area of this category
        | "Hounsfield Unit": the average hounsfield density of this category
    | <CATEGORY2>
    ...
<MODEL2_NAME>
    | ...
```

### Running `cli.py` as a module
Instead of `python abctseg/cli.py`, you can also run `python -m abctseg.cli` 
to use abctseg as a module if you have followed
the [installation instructions](install.html).