#!/bin/bash
# File: feature_extraction.sh
# Author: MLun Wong (lun.m.wong@cuhk.edu.hk)
# Date: 2024-01-29
# Description:
#   This file is the script that make use of the mradtk to extract radiomics features. Please use the docker
#   container if you wish to use this script. There are several steps you need to take:
#     1. Download the original images released by Aerts et al. (and modify the ENV)
#     2. Sample rotations for the downloaded images (see 0_sampling_rotations.ipynb)
#        (Optional) Alternatively, you can  request the rotation transform we sampled and apply it youself.
#     3. Modify the paths specified here.
#     4. Run this script.

export OUTDATED_IGNORE=1

IMAGE_DIR=/media/storage/Data/NSCLC/10.WaveletStudyData/A.ImgPatches
SEGMENT_DIR=/media/storage/Data/NSCLC/10.WaveletStudyData/B.SegPatches
PARAMETER_FILE=/media/storage/Data/NSCLC/10.WaveletStudyData/radiomics_config.yaml
ID_GLOBBER="LUNG1-\d+(-\w)?_R[\d]+"
OUTPUT_FILE="RadFeatures_raw_fine_res.h5"

mradtk-extract-features -i $IMAGE_DIR -s $SEGMENT_DIR -p $PARAMETER_FILE -v -g $ID_GLOBBER -o "$OUTPUT_FILE" -k
