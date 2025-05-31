# Colon Reconstruction: Non-Rigid Registration

We perform non-rigid (B-spline) registration of colon segmentations using [SimpleITK](https://simpleitk.readthedocs.io/). The goal is to align a fixed (collapsed) colon image with the best-matching non-collapsed mask (moving image) and apply the transformation to its colon segmentation.

## Usage

1. **Adjust paths:**  
   Edit the script to set the correct paths for your fixed image, moving image directories, and segmentation directory.  
   The regex pattern for file matching may need to be adapted to your file naming convention.

2. **Run the script:**  
   Execute the script on your HPC or local machine with the necessary libraries installed.

   ```bash
   python bspline_automated.py
   ```
