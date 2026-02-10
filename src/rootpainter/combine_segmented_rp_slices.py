import SimpleITK as sitk
import numpy as np
import os
from PIL import Image


def combine_all_subjects(slices_folder, output_folder):
    """
    Combines PNG slices for each subject in the folder and saves as .mha files.
    Output files are named colon_XXX_combined_fluid.mha, where XXX is the subject number.
    """
    slice_files = [f for f in os.listdir(slices_folder) if f.endswith(".png")]
    if not slice_files:
        print("No PNG slices found in the folder.")
        return

    # Group files by subject (extract subject number from filename)
    subject_dict = {}
    for fname in slice_files:
        # Example: sub002_pos-prone_scan-1_conv-sitk_masked_slice268.png
        if fname.startswith("sub"):
            subj_part = fname.split("_")[0]  # sub002
            subj_num = subj_part[3:]  # 002
            if subj_num not in subject_dict:
                subject_dict[subj_num] = []
            subject_dict[subj_num].append(fname)

    os.makedirs(output_folder, exist_ok=True)

    for subj_num, files in subject_dict.items():
        # Sort by slice index
        def get_slice_index(filename):
            parts = filename.split("_slice")
            if len(parts) > 1:
                idx = parts[1].split(".")[0]
                return int(idx)
            return -1

        files.sort(key=get_slice_index)

        slices = []
        for fname in files:
            img = Image.open(os.path.join(slices_folder, fname)).convert(
                "L"
            )  # Convert to grayscale
            arr = np.array(img)
            # Print unique pixel intensities for this slice
            unique_vals = np.unique(arr)
            print(
                f"Subject {subj_num}, slice {fname}: unique pixel intensities: {unique_vals}"
            )
            # Binarize: everything >0 becomes 1, else 0
            arr_bin = (arr > 0).astype(np.uint8)
            slices.append(arr_bin)
        volume = np.stack(slices, axis=0)

        sitk_img = sitk.GetImageFromArray(volume)
        out_name = f"colon_{subj_num}_combined_fluid.mha"
        out_path = os.path.join(output_folder, out_name)
        sitk.WriteImage(sitk_img, out_path)
        print(f"Saved {out_path} for subject {subj_num}")


if "__main__":
    combine_all_subjects("segmentations", "combined_slices")
