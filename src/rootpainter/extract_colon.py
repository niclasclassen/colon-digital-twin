import SimpleITK as sitk
import numpy as np
import os
import scipy.ndimage as ndimage
import gzip
import shutil


def gunzip_file(gz_path, out_path):
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_path


def mask_colon(raw_img_path, totalsegmentator_mask_path):
    # read images
    raw_image = sitk.ReadImage(raw_img_path)
    unzipped_path = gunzip_file(totalsegmentator_mask_path, "temp_mask.mha")
    totalsegmentator_mask = sitk.ReadImage(unzipped_path)

    raw = sitk.GetArrayFromImage(raw_image)
    totalsegmentator_mask = sitk.GetArrayFromImage(totalsegmentator_mask)

    # perform dilation on total segmentator colon mask
    kernel_size = (3, 3, 3)
    structuring_element = np.ones(kernel_size)

    eroded_mask_np = ndimage.binary_dilation(
        totalsegmentator_mask, structure=structuring_element
    )

    # remove everything outside of eroded_mask_np
    raw_image_masked_np = np.where(eroded_mask_np, raw, 0)

    # convert back to SimpleITK image
    raw_image_masked = sitk.GetImageFromArray(raw_image_masked_np)
    raw_image_masked.CopyInformation(raw_image)

    # save image
    filename = os.path.splitext(os.path.basename(raw_img_path))[0] + "_masked.mha"
    sitk.WriteImage(raw_image_masked, f"masked_colon_rp/{filename}")


def batch_mask_colon(raw_dir, mask_dir):
    for sub in os.listdir(mask_dir):
        sub_path = os.path.join(mask_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        for pos in ["prone", "supine"]:
            sub_num = sub.replace("sub", "")
            sub_formatted = f"{int(sub_num):04d}"
            # raw_name = f"colon_{sub_formatted}-{pos}.mha"
            raw_name = f"{sub}_pos-{pos}_scan-1_conv-sitk.mha"
            mask_name = f"{sub}_pos-{pos}_scan-1_conv-sitk/{sub}_pos-{pos}_scan-1_conv-sitk_totalseg-colon.mha.gz"

            raw_path = os.path.join(raw_dir, raw_name)
            mask_path = os.path.join(sub_path, mask_name)

            if os.path.exists(raw_path) and os.path.exists(mask_path):
                print(f"Processing {raw_path} with {mask_path}")
                try:
                    mask_colon(raw_path, mask_path)
                except Exception as e:
                    print(f"Error processing {raw_path} with {mask_path}: {e}")
            else:
                print(f"Missing file for {sub} {pos}: {raw_path} or {mask_path}")


def main():
    raw_dir = "datasets_raw"
    mask_dir = "segmentations-totalsegmentator"
    batch_mask_colon(raw_dir, mask_dir)


if __name__ == "__main__":
    main()
