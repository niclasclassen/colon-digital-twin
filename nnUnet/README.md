# nnUNet Pipeline Usage

This directory contains scripts to preprocess data, train a model, and run inference using nnUNet on a High-Performance Computing (HPC) environment.

## Pipeline Steps

1. **Preprocessing**

   Run the preprocessing script to prepare your dataset and verify its integrity:

   ```bash
   bash run_preprocessing.sh
   ```

2. **Training**

   Start model training with:

   ```bash
   bash run_training.sh
   ```

3. **Inference**

   After training, run inference on new data:

   ```bash
   bash run_inference.sh
   ```

## Notes

- This pipeline is designed for use on an HPC system.
- Update all placeholder paths and dataset numbers in the scripts before running.
- The pipeline uses the `3d_fullres` configuration.
- Ensure all required environment variables are set as shown in the scripts.

For more details, see comments within each script.

## Reference

For more information about nnUNet, see the [official nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/tree/master).
