import os
import subprocess
from carbontracker.tracker import CarbonTracker

# Set up CarbonTracker for monitoring energy usage
tracker = CarbonTracker(epochs=1)
tracker.epoch_start()

# Define parameters for nnUNet training
NNUNET_TRAIN_BIN = "/path/to/nnUNetv2_train"  # Path to nnUNetv2_train executable
DATASET_NUMBER = "999"  # Dataset number
CONFIG = "3d_fullres"  # Model configuration
FOLD = "0"  # Fold number (0-4)

# Construct the training command with placeholders
command = (
    f"NNUNET_NUM_DATALOADER_WORKERS=0 "
    f"nnUNet_compile=False "
    f"{NNUNET_TRAIN_BIN} {DATASET_NUMBER} {CONFIG} {FOLD} --npz --c"
)

print(">> Starting nnUNet Training with CarbonTracker...", flush=True)
process = subprocess.Popen(command, shell=True, env=os.environ)
process.wait()

# End CarbonTracker monitoring
tracker.epoch_end()
print(">> Training completed and CarbonTracker finished.", flush=True)
