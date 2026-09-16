#!/bin/bash

# Set required environment variables for nnUNet
export nnUNet_raw="./raw"                        # Base directory for raw dataset
export nnUNet_preprocessed="./preprocessed"      # Directory for preprocessed data
export nnUNet_results="./results"                # Directory to store trained model results
export dataset_name="Dataset999_Colon"           # Dataset name
export dataset_number="999"                      # Dataset number
#export fold=1                                   # Fold number (not needed as we use 5-fold cross-validation)

# Print the environment variables for debugging
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "dataset: $dataset_name"
echo "dataset number: $dataset_number"

# Run nnUNet inference (update the placeholders as needed)
nnUNetv2_predict \
  -i ../data/raw \
  -o ../data/predictions \
  -d $dataset_number \
  -c 3d_fullres \
  --verbose \
  --save_probabilities

# -i: Input directory with images to predict on
# -o: Output directory for predictions
# -d: Dataset number (should match training)
# -c: Model configuration (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)
# --verbose: Enables detailed logging output
# --save_probabilities: Saves probability maps (not just final segmentations)


