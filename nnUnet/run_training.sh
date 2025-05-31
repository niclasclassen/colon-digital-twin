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

# Start training the nnUNet model with CarbonTracker monitoring
python training_config.py

# Notes:
# - Set <DATASET_NUMBER> to your dataset number (e.g., 999)
# - Set <CONFIG> to your model configuration (e.g., 3d_fullres)
# - Edit train_with_carbontracker.py to match your training setup