##!/bin/bash

# Set the required environment variables for nnUNet
export nnUNet_raw="./nnunet_raw"                        # Base directory for raw dataset
export nnUNet_preprocessed="./nnunet_preprocessed"      # Directory for preprocessed data
export nnUNet_results="./nnunet_results"                # Directory to store trained model results

# Print the environment variables for debugging
echo "Current working directory: $(pwd)"
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"

# Run preprocessing and verify dataset integrity (using 3d_fullres configuration)
nnUNetv2_plan_and_preprocess -d <DATASET_NUMBER> --verify_dataset_integrity -c 3d_fullres

# -d <DATASET_NUMBER>: Dataset number (e.g., 999)
# --verify_dataset_integrity: Checks for inconsistencies in the dataset format
# -c 3d_fullres: Use the 3D full-resolution model configuration

