# Introduction

This repo consist of code used for ther radiomic analysis of non-small-cell lung cancer (NSCLC) patients' CT scans. In summary, we augment the images by inducing rotations of different degrees and see how the radiomic features values changes. An ideal feature should be identical in its values before and after the rotation. The study further dig into the difference in behavior of wavelet-deposition (WD) derived features and non-WD derived features. 

## Submission to ECR2024

The results in this work was submited to European Congress of Radiology 2024 (ECR2024) in an abstract entitled: *Robustness of Wavelet Features for Radiomics can be Impaired for Lesions without Standard Orientation References*. The abstract was accepted for poster presentation in the venue. 

## Authors & Affiliations

WONG, Lun M.[1]; AI, Qi-yong Hemis[1,2]; HUNG, Kuo Feng[3]; SO, Tiffany YT[1]; KING, Ann[1]

1. Department of Imaging and Interventional Radiology, Prince of Wales Hospital, The Chinese University of Hong Kong, HKSAR
2. Department of Health Technology and Informatics, The Hong Kong Polytechnic University, HKSAR
3. Applied Oral Sciences & Community Dental Care, Faculty of Dentistry, The University of Hong Kong

# Code Documentation

Required packages:
------------------
* numpy
* scipy
* SimpleITK
* pandas
* mnts (https://github.com/alabamagan/mri_normalization_tools)

## Image intensity profile normalization

We normalized the images based on the foreground-tissue mean and variance prior to feature extraction using an in-house developed python package heavily utilized `SimpleITK` to perform normalization, but with the aim to facilitate repeatability. Even though it was originally written for MRI, it can be flexibly configured to include only relevant steps. See [mnts](https://github.com/alabamagan/mri_normalization_tools) for details.


### Normalization of CT scans
```python 
import os
import mnts
import SimpleITK as sitk
import matplotlib.pyplot as plt

from pathlib import Path
from mnts.filters import mpi_wrapper
from mnts.utils import repeat_zip
from mnts.filters.data_node import DataNode, TypeCastNode
from mnts.filters.intensity import *
from mnts.filters.geom import *
from mnts.filters.mnts_filters_graph import MNTSFilterGraph
from mnts.utils.filename_globber import *
from mnts.mnts_logger import MNTSLogger

'''Configurations'''
rootdir = Path(".").absolute()
sourcedir = rootdir.joinpath("./0A.NIFTI_ALL/").absolute()
segdir = rootdir.joinpath("0B.SEGMENT_ALL/GTV-1").absolute()
imgdir = rootdir.joinpath("01.Images").absolute()
outdir = rootdir.joinpath("02.Normalized_Images/ZScore")
outdir.mkdir(exist_ok=True, parents=True)
outmaskdir = rootdir.joinpath("02.Normalized_Images/ForegroundMask")
outmaskdir.mkdir(exist_ok=True, parents=True)
idregpat = "^[\d\w]+-\d+" # for LUNG1-[Number] format

# Set up normalization graph
norm_graph = MNTSFilterGraph()
norm_graph.add_node(DataNode())
norm_graph.add_node(OtsuThresholding(), 0, is_exit=True)
norm_graph.add_node(ZScoreNorm(), [0, 1], is_exit=True)
norm_graph.plot_graph()

# show graph for normalization
mdprint("# Normalization graph")
plt.show()

# Handle both image and segmentation in pairs
pids = get_unique_IDs([str(r.name) for r in imgdir.glob("*nii.gz")],globber=idregpat)
pids.remove('LUNG1-128') # this case has no matching segmentation and is excluded in this study
img_path, seg_path = load_supervised_pair_by_IDs(imgdir, segdir, globber=idregpat, idlist=pids)

# sanity test normalizing one case
res = norm_graph.execute(sitk.ReadImage(imgdir.joinpath(img_path[0]))) # output from node 2

# Save the resutls
normed_img = res[2] # sitk.Image
foreground_mask = res[1] # sitk.Image

# Run a for loop to handle all cases
outdir.mkdir(exist_ok=True, parents=True)
for _img_p in tqdm(img_path):
    MNTSLogger['Normalization'].info(f"Handling {_img_p}")
    try:
        _img = sitk.ReadImage(imgdir.joinpath(_img_p))
        _output_path = outdir.joinpath(_img_p)
        _outputmask_path = outmaskdir.joinpath(_img_p)
        res = norm_graph.execute(_img) # output from node 2
        normed_img = res[2]
        foreground_mask = res[1]
        sitk.WriteImage(normed_img, str(_output_path))
        sitk.WriteImage(foreground_mask, str(_outputmask_path))
    except Exception as e:
        msg = f"Error occurred when handling {_img_p}
        MNTSLogger['Normalization'].warning(msg)
        # also log the exception
        MNTSLogger['Normalization'].exception(e)

```

## Sampling Rotation

Orientation variations with a mean displacement of $\{10\degree,20\degree,40\degree\}$ was introduced by first applying a yaw rotation and then a pitch rotation.The yam rotations was sampled from the normal distributions $\mathscr{N}(0, [10\degree,20\degree,40\degree])$,akin to the shake of the head; the pitch rotations were sampled from $\mathscr{N}([10\degree,20\degree,40\degree],10\degree)$,akin to the node of the head.

```python
from feature_robustness_analysis.rot_ops import gen_rot_set

size = 32 # number of rotations to sample
pitch_mu = deg # pitch distribution mean
pitch_sigma = 10 # pitch distribution std, this is fixed to 10 degrees
yaw_mu = 0 # yaw distribution mean
yaw_sigma = deg # yaw distribution std

# samples the roation set
R = gen_rot_set(size, pitch_mu, pitch_sigma, yaw_mu, yaw_sigma)

# R = [rot1, rot2, ..., rot32]

```

For details, please see [sampling_rotations.ipynb](./sampling_rotations.ipynb)

### Naming system

After sampling and applying the rotations the CT scan patches are named based on the following convention:

`[Patient ID]-[Rotation]-[LesionCode].nii.gz`

## Extracting radiomic features

Radiomic features were extracted using the well-established Pyradiomics (v3.0.0) package. The extracted features are provided within this repository [here](./RadFeatures_raw_ECR2024.xlsx). The yaml file configuration used for feature extraction is given as well in `[root]/pyradiomics_config`


## Data Analysis

Please see [FeatureAnalysis_ECR2024.ipynb](./FeatureAnalysis_ECR2024.ipynb)

## Statistics

Spearman's rank test was conducted using implementation in the package `scipy`