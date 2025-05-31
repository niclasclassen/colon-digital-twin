import os
import glob
import numpy as np
import SimpleITK as sitk


def is_identity_direction(direction):
    return np.allclose(direction, np.eye(3).flatten())


def direction_to_orientation_string(direction_matrix):
    direction_matrix = np.array(direction_matrix).reshape(3, 3)
    labels = [["R", "L"], ["A", "P"], ["I", "S"]]  # positive, negative for each axis

    orientation = ""
    for axis in range(3):  # for each axis row (x, y, z)
        dir_vector = direction_matrix[axis]
        max_idx = np.argmax(np.abs(dir_vector))
        sign = np.sign(dir_vector[max_idx])
        label = labels[max_idx][0] if sign > 0 else labels[max_idx][1]
        orientation += label

    return orientation


def reorient_array_to_rai(image_array, current_orientation):
    """
    Reorients a 3D image array from current_orientation to RAI.

    Parameters
    ----------
    image_array : np.ndarray
        3D image in (z, y, x) order.
    current_orientation : str
        Three-letter string like 'RPS', 'LPS', etc.

    Returns
    -------
    np.ndarray
        Reoriented image array in RAI orientation.
    """
    orientation_map = {
        "R": (1, 0),
        "L": (-1, 0),
        "A": (1, 1),
        "P": (-1, 1),
        "I": (1, 2),
        "S": (-1, 2),
    }

    # Build transformation matrix from current to RAI
    transform = np.eye(3)
    for i, desired in enumerate("RAI"):
        current = current_orientation[i]
        flip_sign, axis = orientation_map[current]
        desired_sign, _ = orientation_map[desired]
        transform[i, axis] = desired_sign * flip_sign

    # Transpose to (x, y, z) for proper spatial remapping
    image_xyz = np.transpose(image_array, (2, 1, 0))
    shape = image_xyz.shape
    reoriented = np.zeros_like(image_xyz)

    coords = (
        np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        .reshape(3, -1)
        .astype(np.float32)
        .T
    )
    new_coords = np.dot(coords, transform.T).astype(np.int32)

    # Adjust for any negative indices due to flipping
    new_coords -= new_coords.min(axis=0)
    new_shape = new_coords.max(axis=0) + 1

    # Clip to stay in bounds
    mask = (
        (new_coords[:, 0] < new_shape[0])
        & (new_coords[:, 1] < new_shape[1])
        & (new_coords[:, 2] < new_shape[2])
    )

    reoriented[new_coords[mask, 0], new_coords[mask, 1], new_coords[mask, 2]] = (
        image_xyz[
            coords[mask, 0].astype(int),
            coords[mask, 1].astype(int),
            coords[mask, 2].astype(int),
        ]
    )

    # Return to (z, y, x)
    return np.transpose(reoriented, (2, 1, 0))


def preprocess_image(
    array_rai, sitk_image, new_spacing=(1.0, 1.0, 1.0), target_shape=(512, 512, 512)
):
    """
    Resamples the input array to the specified spacing and resizes it to the target shape.

    Parameters:
        array_rai (np.ndarray): Original image array (z, y, x).
        sitk_image (sitk.Image): The SimpleITK image (used for original spacing/origin/direction).
        new_spacing (tuple): Desired spacing (e.g., (1.0, 1.0, 1.0)).
        target_shape (tuple): Desired output shape (z, y, x), e.g., (512, 512, 512).

    Returns:
        np.ndarray: Preprocessed image array of shape target_shape.
    """

    def resample_array_to_spacing(array, image, new_spacing):
        original_spacing = image.GetSpacing()
        print("Original spacing:", original_spacing)
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetDefaultPixelValue(image.GetPixelIDValue())

        resampled_img = resample.Execute(image)
        return resampled_img

    # Resample to uniform spacing
    img_resampled = resample_array_to_spacing(array_rai, sitk_image, new_spacing)
    array_resampled = sitk.GetArrayFromImage(img_resampled)  # (z, y, x)

    # # Resize to target shape
    # current_shape = array_resampled.shape
    # zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]

    # array_resized = scipy.ndimage.zoom(
    #     array_resampled, zoom=zoom_factors, order=1
    # )  # Linear interpolation

    return array_resampled


input_folder = "input_folder_path"  # Replace with your input folder path
output_folder = "output_folder_path"  # Replace with your output folder path
os.makedirs(output_folder, exist_ok=True)

file_list = glob.glob(os.path.join(input_folder, "*.mha"))

for full_path in file_list:
    try:
        print(f"Processing {full_path}...")

        # Load image
        sitk_image = sitk.ReadImage(full_path)
        ori_str = direction_to_orientation_string(sitk_image.GetDirection())
        print("Original orientation:", ori_str)

        # Convert to RAI
        array = sitk.GetArrayFromImage(sitk_image)
        # Only reorient if necessary
        if ori_str != "RAI":
            array_rai = reorient_array_to_rai(array, ori_str)
            print(f"Reoriented from {ori_str} to RAI.")
        else:
            array_rai = array
            print("Already in RAI orientation. No reorientation needed.")

        # Normalize spacing and dimensions
        array_resized = preprocess_image(array_rai, sitk_image)

        # Convert back to SimpleITK Image
        img = sitk.GetImageFromArray(array_resized)
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetDirection(np.eye(3).flatten())
        img.SetOrigin((0.0, 0.0, 0.0))  # Optional: reset origin if needed

        # Save the processed image
        filename = os.path.basename(full_path)
        output_path = os.path.join(output_folder, filename)
        sitk.WriteImage(img, output_path)

        print(f"Saved processed file to {output_path}\n")

    except Exception as e:
        print(f"Failed to process {full_path}: {e}\n")

print("Batch processing completed.")
