# Semi-Automatic Colon Label Generation

Our method combines image processing, morphological operations, and anatomical heuristics to segment colon masks and remove unwanted structures (e.g., air, bed, lungs).

## Features

- Removes surrounding air, bed, and lungs from CT images.
- Uses shape, position, and overlap heuristics to distinguish colon from other structures.
- Merges colon and fluid regions for improved segmentation.
- Designed run on a HPC.

## Usage

1. **Adjust paths:**  
   Edit the script to set the correct paths for your raw images, segmentations, and exclusion list.

2. **Run the script:**  
   Execute the script with Python 3 and required libraries installed:

   ```bash
   python segment_colon_and_fluid.py
   ```
