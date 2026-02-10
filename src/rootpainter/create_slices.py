import SimpleITK as sitk
import numpy as np
import os
from PIL import Image


def extract_and_save_slices(mha_file_path, output_dir, slice_range=5):
    """
    Extracts slices in the axial plane from a .mha file and saves them as high-quality JPEG files.

    Parameters:
        mha_file_path (str): Path to the .mha file.
        output_dir (str): Directory to save the extracted slices as JPEGs.
        slice_range (int): Number of slices around the center to extract.
    """
    # Load the .mha file
    image = sitk.ReadImage(mha_file_path)

    # Get the 3D image array
    image_array = sitk.GetArrayFromImage(image)

    # Extract the base name for the file without extension
    base_name = os.path.basename(mha_file_path).split(".mha")[0]

    # Calculate the center slice index in the axial plane
    center_index = image_array.shape[0] // 2

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract slices around the center and save as JPEG
    # start_slice = max(center_index - slice_range // 2, 0)
    # end_slice = min(center_index + slice_range // 2 + 1, image_array.shape[0])

    for i in range(image_array.shape[0]):
        slice_image = image_array[i, :, :]

        # Normalize the slice to the range [0, 255]
        slice_image = (
            (slice_image - np.min(slice_image))
            / (np.max(slice_image) - np.min(slice_image))
            * 255
        ).astype(np.uint8)

        # Convert to a PIL Image
        pil_image = Image.fromarray(slice_image)

        # Resize image to 750 x 750
        pil_image = pil_image.resize((750, 750), Image.Resampling.LANCZOS)

        # Save as high-quality JPEG
        # output_file = os.path.join(output_dir, f"slice_{i:03d}.jpeg")
        output_file = os.path.join(output_dir, f"{base_name}_slice{i:03d}.jpeg")
        pil_image.save(output_file, "JPEG", quality=95)

        print(f"Saved slice {i} as {output_file}")


def extract_and_save_slices_from_directory(
    input_dir, output_dir, slice_range=100, max_files=None
):
    """
    Extract slices from a limited number of .mha files in a directory and its subdirectories.

    Parameters:
        input_dir (str): Directory containing .mha files.
        output_dir (str): Directory to save the extracted slices.
        slice_range (int): Number of slices around the center to extract.
        max_files (int): Maximum number of .mha files to process. If None, process all files.
    """
    file_count = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mha"):
                if max_files is not None and file_count >= max_files:
                    print("Reached the maximum number of files to process.")
                    return
                mha_file_path = os.path.join(root, file)
                print(f"Processing file: {mha_file_path}")
                extract_and_save_slices(mha_file_path, output_dir, slice_range)
                file_count += 1


# mha_file_path = "/home/martina/Dataset/IRE_masked_images/"
input_dir = "masked_colon_rp"
output_dir = "split_files"

# Extract from directory
max_files = 400  # Change this to the number of files you want to process
extract_and_save_slices_from_directory(input_dir, output_dir, max_files=max_files)

# Extract from single file
# extract_and_save_slices(mha_file_path, output_dir)
