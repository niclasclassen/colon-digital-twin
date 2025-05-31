import os
import shutil
import re
import gzip

# Define paths
SOURCE_FOLDER = "source_colon_data"  # Replace with the actual source folder path
DEST_FOLDER = "formatted_colon_data"  # Replace with the desired destination folder path

# Ensure the destination folder exists
os.makedirs(DEST_FOLDER, exist_ok=True)

# Loop through subject folders
for subject_folder in sorted(os.listdir(SOURCE_FOLDER)):
    subject_path = os.path.join(SOURCE_FOLDER, subject_folder)
    subject_number = re.search(r"(\d+)", subject_folder)

    if not subject_number:
        print(f"Skipping {subject_folder}: No subject number found.", flush=True)
        continue
    subject_number = subject_number.group(1)

    if not os.path.isdir(subject_path):
        continue  # Skip if it's not a directory

    supine_file_path = (
        f"{subject_path}/sub{subject_number}_pos-supine_scan-1_conv-sitk.mha.gz"
    )
    prone_file_path = (
        f"{subject_path}/sub{subject_number}_pos-prone_scan-1_conv-sitk.mha.gz"
    )

    # check if at least one of the files exists
    if not (os.path.exists(supine_file_path) or os.path.exists(prone_file_path)):
        print(
            f"Warning: Neither supine nor prone file exists for {subject_folder}. Skipping...",
            flush=True,
        )
        continue

    new_subject_num = f"{int(subject_number):04d}"  # Format subject number as 4-digit

    # Rename, unzip, and copy supine file if it exists
    if os.path.exists(supine_file_path):
        supine_new_name = f"colon_{new_subject_num}-supine_0000.mha"
        supine_dest_path = os.path.join(DEST_FOLDER, supine_new_name)
        with gzip.open(supine_file_path, "rb") as f_in:
            with open(supine_dest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"Warning: Supine file missing for {subject_folder}.", flush=True)

    # Rename, unzip, and copy prone file if it exists
    if os.path.exists(prone_file_path):
        prone_new_name = f"colon_{new_subject_num}-prone_0000.mha"
        prone_dest_path = os.path.join(DEST_FOLDER, prone_new_name)
        with gzip.open(prone_file_path, "rb") as f_in:
            with open(prone_dest_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"Warning: Prone file missing for {subject_folder}.", flush=True)

    print(
        f"Processed subject {new_subject_num}: supine and prone files copied.",
        flush=True,
    )

print("Processing complete.", flush=True)
