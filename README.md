# Info

This folder contain the juptyer notebooks used in this study. There are four notebooks in total which was setup based on their purpose.

- [Info](#info)
- [Setting up environment](#setting-up-environment)
- [Notebooks](#notebooks)
  - [\[0\] Sampling rotations](#0-sampling-rotations)
    - [Sampling rotational matrix](#sampling-rotational-matrix)
      - [Usage](#usage)
    - [Applying rotational matrix](#applying-rotational-matrix)
      - [Usage](#usage-1)
  - [\[0\] Feature extraction](#0-feature-extraction)
    - [Usage](#usage-2)
  - [\[1\] Feature-level analysis](#1-feature-level-analysis)
    - [Statstical test](#statstical-test)
    - [Usage](#usage-3)
  - [\[2\] Model training and testing](#2-model-training-and-testing)
    - [Usage](#usage-4)
  - [\[3\] Performance-level analysis](#3-performance-level-analysis)


# Setting up environment

A `Dockerfile` is prepared to build a docker image for running the code. To build an image, you can use the command:

```bash
docker build --rm -f "Dockerfile" -t radiomicsanalysis:latest "." 
```

Test the function of the package by going into `src/tests` and run `pytest`:

```bash
pytest .
```

You should excpect all test to pass. 

# Notebooks

The code used in this study are provided with the package `feature_robustness_analysis` as well as in the jupyter notebook under `notebooks`. Note that the notebooks depend on the `feature_robustness_analysis` package and you are advised to use the dockerfile to create the environment for testing. 

There are 4 notebooks corresponding to different major part of the study:

## [0] Sampling rotations

The jupyter notebook `0_sampling_rotations.ipynb` contains the code for preprocessing the CT images. First, rotations were sampled into for each connected components of the NSCLC segmentation. Second, the rotations were applied to the center-of-mass of the corresponding components using `sitk` resampling function. The resampling will also crop the image to the size of the connected component. Finally, the cropped pactch is saved to an output directory with naming format `[SID]_[LesionCode].nii.gz` (Lesion code is for patients with multiple segmented components).

### Sampling rotational matrix

To sample rotational matrix, you can use the function written in package `feature_robustness_analysis`. An example is as follow:

#### Usage

```python
from feature_robustness_analysis.rot_ops import *

# Load images as a list
imsets = ... 
coms = ...

rots = gen_rot_set(size=len(imsets), pitch_mu=10, pitch_sigma=10, yaw_mu=0, yaw_sigma=10)
# List of scipy `transform.Rotation` objects
rots = [rot_to_affine(rr, cents) for rr, cc in zip(imsets, coms)]
# List of `sitk.AffineTransform`
```

### Applying rotational matrix

The rotations can be applied using functions in `feature_robustness_analysis.im_ops`. The main functions used were `get_connected_bodies`, `resample_to_segment_bbox` and `resample_image`. See the following snippet copied from the main notebook:

#### Usage

```python
from feature_robustness_analysis.im_ops import *
import SimpleITK as sitk

# Calculate number of lesions, COM of lesions here
...

# Load images
_img = sitk.ReadImage(str(img_dir.joinpath(_img_p)))
_seg = sitk.ReadImage(str(seg_dir.joinpath(_seg_p)))
num_of_lesions = df.loc[_pid]['Number of lesions']
if num_of_lesions > 1:
    _seg_conn = get_connected_bodies(_seg)
    if len(_seg_conn) != num_of_lesions:
        raise IndexError(f"Number of segmentation components specified is incorrect for {_pid}: {_seg_p}")
    for i, _sc in enumerate(_seg_conn):
        # Obtain the sampled affine transform
        _transform = rotations[deg][_new_pid]
        # Apply the transform by resampling
        _img_out = resample_image(_img, _transform)
        _seg_out = resample_image(_sc, _transform)
        # Crop the resampled image to the bounding box
        _img_out, _seg_out = resample_to_segment_bbox(_img_out, _seg_out, padding=10)
  
        # Write image 
        sitk.WriteImage(_img_out, str(_imgfname))
        sitk.WriteImage(_seg_out, str(_segfname))

...
```

## [0] Feature extraction

Feature extraction was simply done using a bash script under an environment with the necessary python package `mradtk` installed. The pacakge can be found [here](https://github.com/alabamagan/mri_radiomics_toolkit/tree/pre_release) The script can be found in the file `feature_extraction.sh`, a copy of usage is given below:

### Usage

```bash
#!/bin/bash
export OUTDATED_IGNORE=1 # This mute some warnings

IMAGE_DIR=/media/storage/Data/NSCLC/10.WaveletStudyData/A.ImgPatches
SEGMENT_DIR=/media/storage/Data/NSCLC/10.WaveletStudyData/B.SegPatches
PARAMETER_FILE=/media/storage/Data/NSCLC/10.WaveletStudyData/radiomics_config.yaml
ID_GLOBBER="LUNG1-\d+(-\w)?_R[\d]+"
OUTPUT_FILE="RadFeatures_raw_fine_res_v3.xlsx"

mradtk-extract-features -i $IMAGE_DIR -s $SEGMENT_DIR -p $PARAMETER_FILE -v -g $ID_GLOBBER -o "$OUTPUT_FILE" -k

```

> Notes: Follow the instruction on the mradtk repo for installation to get the `mradtk-extract-features` command.

## [1] Feature-level analysis

The jupyter notebook `1_feature_analysis.ipynb` contains the code that was written for feature analysis and visualization of the results. Most of the details were covered in the maintext and the supplementary material of the original paper. The jupyter notebook also contians more detailed descriptions of the code.

### Statstical test

For feature analysis, the statistical tests were conducted using the statistical package offered by `scipy` package. Specifically, the paired-t-tests were conducted using `scipy.stat.ttest_rel` and the independent sampled t-test were conducted using `scipy.stat.ttest_ind`. Additionally, the Spearman's rank test analysis was conducted using functions offered by `scipy.stat.spearmanr`.

Prior to running the script, we have verified the results of the `scipy.stat.ttest_rel` and `scipy.stat.ttest_ind` on a small subset of all features to ensure the results were correctly calculated.

The python script for this part of analysis, apart from those on the jupyter notebook, can also be found in the `feature_robustness_anlaysis` package. Specifically, [`feature_robustness_analysis/stat.py`](./feature_robustness_analysis/stats.py).

### Usage

```python
# dataframe of features 
# comparison is always done between each column (R_5 to R_80) to the first column (R_0).
df = ...

# Returns a df with two columns p-val and Spearman's CC.
identify_trend_from_df(df)

```

This notebook and code package can be run within the docker container build with the give Docker file.


## [2] Model training and testing

The jupyter notebook [`2_model_training.ipynb`](notebooks/2_model_training.ipynb) contains the code implementation of the radiomic pipeline, repeated multiple times in a 5-fold cross-validation fashion.

### Usage

This notebook can be run with the docker container created using the docker file.

## [3] Performance-level analysis

The jupyter notebook [`3_performance_analysis.ipynb`](notebooks/3_performance_analysis.ipynb) contains the code implementation for the performance analysis. 

This notebook requires SPSS to be installed and running inorder for the code to run correctly. This is a limitation of the SPSS-python API provided by IBM. There are more description about how the interface can be used. 