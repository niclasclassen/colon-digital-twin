import os
import gzip
import shutil
import SimpleITK as sitk

# Directories
regiongrowing_dir = "path_to_regiongrowing_masks"  # Replace with the actual path to your region-growing masks
fluid_dir = "path_to_fluid_masks"  # Replace with the actual path to your fluid masks
output_dir = "combined_masks"  # Directory to save the combined masks

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


def decompress_gz(file_path):
    """Decompress a .gz file and return the decompressed file path."""
    if not file_path.endswith(".gz"):
        return file_path  # Return the original path if it's not a .gz file

    decompressed_path = file_path[:-3]  # Remove the .gz extension
    with gzip.open(file_path, "rb") as f_in:
        with open(decompressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return decompressed_path


# Iterate through subfolders in the regiongrowing directory
for subfolder in os.listdir(regiongrowing_dir):
    subfolder_path = os.path.join(regiongrowing_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue  # Skip files, only process directories

    # Iterate through files in the subfolder
    for region_file in os.listdir(subfolder_path):
        if not region_file.endswith(".mha.gz"):
            continue  # Skip non-MHA.GZ files

        # Full path to the region-growing mask
        region_file_path = os.path.join(subfolder_path, region_file)

        # Decompress the region-growing mask
        decompressed_region_path = decompress_gz(region_file_path)

        # Extract the subject number and position from the region file
        parts = region_file.split("_")
        subject_id = parts[0]  # Extract the subject ID (e.g., "sub001")
        position = parts[1]
        scan = parts[2]  # Extract the scan (e.g., "scan-1")

        # Skip files where the scan is not "scan-1"
        if scan != "scan-1":
            print(f"Skipping file with scan not equal to 1: {region_file}")
            continue

        fluid_file_name = f"{subject_id}_{position}_scan-1_fluidmask_corrected.mha"
        fluid_file_path = os.path.join(fluid_dir, fluid_file_name)
        print(fluid_file_path, flush=True)

        # Read the region-growing mask
        region_mask = sitk.ReadImage(decompressed_region_path)

        if os.path.exists(fluid_file_path):
            print("Fluid file exists", flush=True)
            # If the fluid mask exists, read it
            fluid_mask = sitk.ReadImage(fluid_file_path)

            # Ensure the masks have the same size and spacing
            if region_mask.GetSize() != fluid_mask.GetSize():
                print(
                    f"Resampling fluid mask to match region-growing mask: {fluid_file_path}"
                )
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(region_mask)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                fluid_mask = resampler.Execute(fluid_mask)

            # Cast both masks to the same pixel type (8-bit unsigned integer)
            region_mask = sitk.Cast(region_mask, sitk.sitkUInt8)
            fluid_mask = sitk.Cast(fluid_mask, sitk.sitkUInt8)

            # Combine the masks (logical OR operation)
            combined_mask = sitk.Or(region_mask, fluid_mask)
        else:
            # If no fluid mask exists, use the region-growing mask as is
            print("File does not exist")
            combined_mask = region_mask

        # Save the combined mask to the output directory
        output_file_name = (
            f"{region_file.replace('_conv-sitk_thr-n800_nei-1.mha.gz', '.mha')}"
        )
        output_file_path = os.path.join(output_dir, output_file_name)
        sitk.WriteImage(combined_mask, output_file_path)

        print(f"Saved combined mask to: {output_file_path}")

        os.remove(decompressed_region_path)
