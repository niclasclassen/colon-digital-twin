from helper import *
import numpy as np
import scipy.ndimage as ndimage
from itertools import product
import os
from skimage.morphology import convex_hull_image, skeletonize
import re
import gzip
import shutil
from skimage.measure import regionprops
from scipy.ndimage import (
    distance_transform_edt,
    binary_fill_holes,
    distance_transform_edt,
    binary_closing,
    binary_propagation,
)
from scipy.ndimage import gaussian_filter


def generate_skeleton(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    binary_mask = binary_fill_holes(binary_mask).astype(np.uint8)
    # Generate the skeleton using 3D thinning
    component_skeleton = skeletonize(binary_mask)
    return component_skeleton


class ColonSegmentation:
    def __init__(
        self,
        raw_image: np.ndarray,
        raw_image_spacing: tuple,
        raw_image_origin: tuple,
        raw_image_direction: tuple,
        colon: np.ndarray,
        lower_lobe_left: np.ndarray,
        lower_lobe_right: np.ndarray,
        middle_lobe_right: np.ndarray,
        upper_lobe_left: np.ndarray,
        upper_lobe_right: np.ndarray,
        small_bowel: np.ndarray,
        air_intensity_threshold: int = -500,
    ):
        # Initialize raw image and totalsegmentator masks
        self.raw_image = raw_image
        self.modified_image = raw_image.copy()

        self.raw_image_spacing = raw_image_spacing
        self.raw_image_origin = raw_image_origin
        self.raw_image_direction = raw_image_direction

        self.lower_lobe_left = lower_lobe_left
        self.lower_lobe_right = lower_lobe_right
        self.middle_lobe_right = middle_lobe_right
        self.upper_lobe_left = upper_lobe_left
        self.upper_lobe_right = upper_lobe_right
        self.colon = colon
        self.small_bowel = small_bowel

        self.air_mask = raw_image < air_intensity_threshold
        self.air_region = np.zeros_like(self.air_mask, dtype=bool)

    def remove_surrounding_air(self):
        # Generate seed points in the corners of the 3D image
        seed_points = list(
            product(
                [0, self.raw_image.shape[0] - 1],
                [0, self.raw_image.shape[1] - 1],
                [0, self.raw_image.shape[2] - 1],
            )
        )

        # Make sure seed points are placed in the air
        seed_points = [seed for seed in seed_points if self.air_mask[seed]]

        labeled_mask, num_features = ndimage.label(self.air_mask)

        # Perform region growing from each seed
        for seed_point in seed_points:
            label_value = labeled_mask[
                seed_point
            ]  # Get the label value at the seed point
            self.air_region |= labeled_mask == label_value  # Add connected air regions

        # Remove external air by setting it to white (max intensity of image)
        self.modified_image[self.air_region] = np.max(self.raw_image)

    def remove_bed(self):
        labeled_mask, num_features = ndimage.label(
            self.air_region == 0
        )  # Identify black regions

        # Compute the size of each component
        component_sizes = np.array(
            ndimage.sum(self.air_region == 0, labeled_mask, range(num_features + 1))
        )

        # Sort components by size (descending order)
        sorted_components = np.argsort(-component_sizes)

        # Identify the second-largest black component
        largest_component = sorted_components[0]  # Background air (should remain)
        second_largest_component = sorted_components[1]  # The unwanted region

        # Create a 3D mask for the second-largest black component
        bottom_object_mask = labeled_mask == second_largest_component

        print(
            "Size of second-largest component:", component_sizes[sorted_components[1]]
        )

        # Create a mask for the second-largest component
        second_largest_mask = (
            labeled_mask == sorted_components[1]
        )  # Second-largest component

        # Remove the second-largest component from the image by setting it to white
        self.modified_image[second_largest_mask] = np.max(self.modified_image)

        print("Bed removed")

    def remove_lungs(self):
        # Combine all lung lobes into one mask
        lungs_mask = (
            (self.lower_lobe_left > 0)
            | (self.lower_lobe_right > 0)
            | (self.middle_lobe_right > 0)
            | (self.upper_lobe_left > 0)
            | (self.upper_lobe_right > 0)
        )

        # Apply the mask to remove lungs from the image
        self.modified_image[lungs_mask] = np.max(self.modified_image)

        print("Lungs removed")

    def check_shape(self, component_mask):
        # Get coordinates of the component's nonzero pixels
        y_indices, x_indices, z_indices = np.where(component_mask)

        if len(y_indices) == 0:
            return 0  # Default to intestine

        # Compute bounding box size
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)

        height = max_y - min_y + 1
        width = max_x - min_x + 1

        elongation_ratio = height / width if width > 0 else 1
        if elongation_ratio > 5.5:
            return 0

        # print("Elongation: ", elongation_ratio)

        solidity_values = []

        # Axial view
        for slice_idx in range(component_mask.shape[0]):
            slice_mask = component_mask[slice_idx, :, :]
            if np.sum(slice_mask) == 0:
                continue
            chull = convex_hull_image(slice_mask)
            solidity_values.append(
                np.sum(slice_mask) / np.sum(chull) if np.sum(chull) > 0 else 0
            )

        # Coronal View
        for slice_idx in range(component_mask.shape[1]):
            slice_mask = component_mask[:, slice_idx, :]
            if np.sum(slice_mask) == 0:
                continue
            chull = convex_hull_image(slice_mask)
            solidity_values.append(
                np.sum(slice_mask) / np.sum(chull) if np.sum(chull) > 0 else 0
            )

        # Sagittal View
        for slice_idx in range(component_mask.shape[2]):
            slice_mask = component_mask[:, :, slice_idx]
            if np.sum(slice_mask) == 0:
                continue
            chull = convex_hull_image(slice_mask)
            solidity_values.append(
                np.sum(slice_mask) / np.sum(chull) if np.sum(chull) > 0 else 0
            )

        avg_solidity = np.mean(solidity_values) if solidity_values else 0

        # print("Solidity: ", avg_solidity)
        strict_solidity = avg_solidity > 0.5

        if strict_solidity:
            return 1  # Colon
        return 0  # Intestine

    def check_position(
        self,
        component_mask,
        colon_intestine_mask,
        radius_factor=0.25,
        spread_threshold=0.3,
    ):
        shape = colon_intestine_mask.shape
        center_z = shape[0] // 2
        center_y = shape[1] // 2
        center_x = shape[2] // 2

        image_width = shape[1]
        radius = image_width * radius_factor

        # Find component points
        comp_x, comp_y, comp_z = np.where(component_mask)
        total_points = len(comp_z)

        # Get number of points within the circle
        distances = np.sqrt(
            (comp_z - center_z) ** 2
            + (comp_x - center_x) ** 2
            + (comp_y - center_y) ** 2
        )
        inside_circle = np.sum(distances < radius)
        total_points = len(comp_y)
        centrality_score = inside_circle / total_points if total_points > 0 else 0

        spread_x = (
            (np.max(comp_x) - np.min(comp_x)) / shape[1] if total_points > 0 else 0
        )
        spread_y = (
            (np.max(comp_y) - np.min(comp_y)) / shape[2] if total_points > 0 else 0
        )
        spread_z = (
            (np.max(comp_z) - np.min(comp_z)) / shape[0] if total_points > 0 else 0
        )

        print("centrality: ", centrality_score)

        if centrality_score > 0.7:  # Likely intestine
            if (
                max(spread_x, spread_y, spread_z) > spread_threshold
            ):  # But spreads widely, so it's likely colon
                return 1  # COLON
            else:
                return 0  # INTESTINE
        return 1

    def merge_colon_fluid_gap(
        self,
        volume,
        colon_mask,
        t_gap_width=2,
        t_grady=-0.98,
        min_area=3,
        min_ecc=0.9,
        max_flatness=0.45,
    ):

        zdim, ydim, xdim = volume.shape
        merged_mask = colon_mask.copy()

        for z in range(zdim):
            slice_img = volume[z]
            air_mask = colon_mask[z]
            fluid_mask = slice_img > 100

            # Distance transforms
            # dt_air = distance_transform_edt(~air_mask)
            # dt_fluid = distance_transform_edt(~fluid_mask)

            dt_air = distance_transform_edt(np.logical_not(air_mask))
            dt_fluid = distance_transform_edt(np.logical_not(fluid_mask))

            # Y-gradient only (vertical axis)
            grad_air_y = np.gradient(dt_air, axis=0)
            grad_fluid_y = np.gradient(dt_fluid, axis=0)
            dot_product_y = grad_air_y * grad_fluid_y

            # Gap condition: close to both air + fluid, gradients oppose in y-dir
            m1 = dt_air < t_gap_width
            m2 = dt_fluid < t_gap_width
            m3 = dot_product_y < t_grady
            raw_gap = m1 & m2 & m3

            # Filter by shape
            labeled_gap, _ = ndimage.label(raw_gap)
            final_gap_mask = np.zeros_like(raw_gap, dtype=bool)
            for region in regionprops(labeled_gap):
                if region.area >= min_area and region.eccentricity > min_ecc:
                    minr, minc, maxr, maxc = region.bbox
                    height = maxr - minr
                    width = maxc - minc
                    flatness = height / (width + 1e-5)
                    if flatness < max_flatness:
                        final_gap_mask[labeled_gap == region.label] = True

            colon_seed = air_mask | final_gap_mask
            grown_fluid = binary_propagation(input=colon_seed, mask=fluid_mask)

            # Final colon region
            merged = colon_seed | grown_fluid
            merged = binary_closing(merged)

            merged_mask[z] = merged

        return merged_mask.astype(np.uint8)

    def segment_colon(self):
        air_mask = self.modified_image < -500

        # Label connected components (air-filled structures)
        labeled_mask, num_features = ndimage.label(air_mask)
        component_sizes = np.array(
            ndimage.sum(air_mask, labeled_mask, range(num_features + 1))
        )

        sorted_components = np.sort(component_sizes)[::-1]
        for i in range(len(sorted_components[:10])):
            print(sorted_components[i], flush=True)

        # Set a threshold to ignore vry small components
        size_threshold = 1000
        filtered_air_mask = np.zeros_like(labeled_mask, dtype=bool)

        # perform dilation on total segmentator masks
        kernel_size = (3, 3, 3)
        structuring_element = np.ones(kernel_size)

        kernel_size_colon = (2, 2, 2)
        structuring_element_colon = np.ones(kernel_size)

        # Apply binary erosion to the colon mask
        eroded_colon_mask = ndimage.binary_erosion(
            self.colon, structure=structuring_element_colon
        )

        # Apply dilation on small bowel mask
        dilated_small_bowel_mask = ndimage.binary_dilation(
            self.small_bowel, structure=structuring_element
        )
        dilated_small_bowel_mask = ndimage.binary_closing(
            dilated_small_bowel_mask, structure=structuring_element
        )

        for label in range(1, num_features + 1):  # Skip background (label 0)
            if component_sizes[label] > size_threshold:
                filtered_air_mask |= labeled_mask == label

        # Re-label only the remaining large components (after removing the very small ones)
        filtered_labeled_mask, new_num_features = ndimage.label(filtered_air_mask)

        # Compute overlap with the dilated colon mask
        overlap_ratios_colon = np.zeros(new_num_features + 1)
        overlap_ratios_small_bowel = np.zeros(new_num_features + 1)
        overlapping_components = 0

        for label in range(1, new_num_features + 1):  # Skip background (label 0)
            component = filtered_labeled_mask == label

            # Calculate intersection with the dilated colon mask
            intersection_colon = np.sum(component & eroded_colon_mask)
            intersection_small_bowel = np.sum(component & dilated_small_bowel_mask)

            # Calculate overlap ratio (intersection / component size)
            component_size = np.sum(component)
            overlap_ratios_colon[label] = (
                intersection_colon / component_size if component_size > 0 else 0
            )
            overlap_ratios_small_bowel[label] = (
                intersection_small_bowel / component_size if component_size > 0 else 0
            )

        # FINAL THINGS
        overlap_threshold = 0.8
        min_votes = 3

        final_colon_mask = np.zeros_like(filtered_labeled_mask, dtype=bool)
        colon_intestine_mask = filtered_labeled_mask > 0

        overlapping_components = 0

        skeleton = generate_skeleton(self.colon)
        for label in range(1, new_num_features + 1):
            component_mask = filtered_labeled_mask == label

            # 1: Overlap Check
            overlap_colon = overlap_ratios_colon[label]
            overlap_small_bowel = overlap_ratios_small_bowel[label]

            if overlap_small_bowel < 0.1:

                overlap_check = int(
                    overlap_colon > overlap_threshold
                    and overlap_colon > overlap_small_bowel
                )

                # if it doesn't pass the overlap check but it's still substatial and it is a large component => colon
                if (
                    overlap_colon > 0.6
                    and not overlap_check
                    and np.sum(component_mask) > 1000000
                ) or overlap_check:
                    score = 1
                else:
                    score = 0

                if overlap_colon > 0.5:
                    # 2: Shape Check
                    shape_check = self.check_shape(component_mask)

                    # 3: Position Check
                    position_check = self.check_position(
                        component_mask, colon_intestine_mask
                    )

                    # 4: Distance to centerline
                    inverse_skeleton = 1 - skeleton
                    # Compute Euclidean Distance Transform (EDT) to nearest skeleton pixel
                    distance_transform = distance_transform_edt(inverse_skeleton)
                    # Keep distances only for pixels within the mask
                    distance_in_mask = distance_transform * component_mask
                    component_size = np.sum(
                        component_mask
                    )  # Count nonzero pixels in the component
                    median_distance = np.median(distance_in_mask[component_mask > 0])
                    normalized_median_distance = median_distance / np.sqrt(
                        component_size
                    )
                    print(normalized_median_distance, flush=True)
                    if normalized_median_distance < 0.1:
                        score += 1

                    score += position_check + shape_check

                if score >= min_votes:

                    print(
                        f"Label: {label} | Score: {score}/4 | Overlap: {overlap_colon:.2f} | Overlap (small bowel): {overlap_small_bowel:.2f}| Shape: {shape_check} | Position: {position_check}",
                        flush=True,
                    )
                    print("Component Size:", np.sum(component_mask))
                    final_colon_mask |= component_mask
                    overlapping_components += 1

        print(
            "Number of components classified as colon:",
            overlapping_components,
            flush=True,
        )

        # Merge colon and fluid gap
        mask_with_fluid = self.merge_colon_fluid_gap(
            self.modified_image, final_colon_mask
        )

        ct_fluid_range = self.modified_image > 200
        mask_with_fluid = binary_propagation(input=mask_with_fluid, mask=ct_fluid_range)

        # filter to remove small irrelevant components
        labeled_mask, _ = ndimage.label(mask_with_fluid)
        filtered_mask = np.zeros_like(mask_with_fluid, dtype=bool)

        for region in regionprops(labeled_mask):
            if region.area >= 1000:
                print(region.area)
                filtered_mask[labeled_mask == region.label] = True

        filtered_mask = filtered_mask.astype(np.uint8)

        # Apply Gaussian filter to smooth the mask
        smoothed = gaussian_filter(filtered_mask.astype(float), sigma=1.0)
        filtered_mask = smoothed > 0.5  # threshold back to binary

        # final_mask = filtered_air_mask | filtered_mask

        # Count components in filtered_air_mask
        _, num_air_components = ndimage.label(filtered_air_mask)

        # Count components in filtered_mask
        _, num_fluid_components = ndimage.label(filtered_mask)

        print(f"Air mask has {num_air_components} components", flush=True)
        print(f"Final mask has {num_fluid_components} components", flush=True)

        # save as mha file
        output_array = filtered_mask.astype(np.uint8)
        output_image = sitk.GetImageFromArray(output_array)

        output_image.SetSpacing(self.raw_image_spacing)  # Copy voxel spacing
        output_image.SetOrigin(self.raw_image_origin)  # Copy origin
        output_image.SetDirection(self.raw_image_direction)  # Copy orientation

        segmentations_folder = (
            "path/to/segmentations"  # Change this to your segmentations folder
        )
        output_path = os.path.join(
            segmentations_folder, f"{subject}_pos-{position}_colon_segmentation.mha"
        )
        sitk.WriteImage(output_image, output_path)


if __name__ == "__main__":

    def load_exclusion_list(path):
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())

    def extract_info(filename):
        # Regex pattern to extract subject ID and position
        pattern = r"colon_\d(\d{3})-([a-zA-Z]+)_"
        match = re.search(pattern, filename)

        if match:
            subject = match.group(1)
            position = match.group(2)
            return subject, position
        return None, None

    def get_files_info(folder_path):
        files_info = []

        for filename in os.listdir(folder_path):
            subject, position = extract_info(filename)
            if subject and position:
                files_info.append((filename, subject, position))

        return files_info

    def open_and_unzip_files(raw_image_path, segmentations_folder, subject, position):
        sub_num = "sub" + str(subject)
        subject_folder = os.path.join(segmentations_folder, sub_num)
        if not os.path.isdir(subject_folder):
            print(f"Subject folder {subject_folder} not found.", flush=True)
            return {}

        # Find the corresponding position folder
        for folder in os.listdir(subject_folder):
            if position in folder:
                position_folder = os.path.join(subject_folder, folder)
                break
        else:
            print(
                f"No matching folder for position {position} in {subject_folder}.",
                flush=True,
            )
            return {}

        # List of expected files
        file_patterns = [
            "totalseg-lung_lower_lobe_left.mha.gz",
            "totalseg-lung_lower_lobe_right.mha.gz",
            "totalseg-lung_middle_lobe_right.mha.gz",
            "totalseg-lung_upper_lobe_left.mha.gz",
            "totalseg-lung_upper_lobe_right.mha.gz",
            "totalseg-colon.mha.gz",
            "totalseg-small_bowel.mha.gz",
        ]

        images = {}

        raw_image = sitk.ReadImage(raw_image_path)
        images["raw_image"] = sitk.GetArrayFromImage(raw_image)
        images["raw_image_spacing"] = raw_image.GetSpacing()
        images["raw_image_origin"] = raw_image.GetOrigin()
        images["raw_image_direction"] = raw_image.GetDirection()

        for pattern in file_patterns:
            expected_file = f"{sub_num}_pos-{position}_scan-1_conv-sitk_{pattern}"
            file_path = os.path.join(position_folder, expected_file)

            if not os.path.exists(file_path):
                print(f"File {file_path} not found.")
                continue

            # Unzipping the .gz file
            unzipped_file_path = file_path[:-3]  # Remove .gz extension
            with gzip.open(file_path, "rb") as f_in:
                with open(unzipped_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Unzipped: {unzipped_file_path}")

            # Read the unzipped file using SimpleITK
            image = sitk.ReadImage(unzipped_file_path)
            images[pattern] = sitk.GetArrayFromImage(image)  # Convert to numpy array

            print(f"Loaded image from {unzipped_file_path}", flush=True)

        return images

    # Example usage
    folder_path = "raw-colons_path"  # Change this to your folder path
    segmentations_folder = (
        "totalsegmentator_folder"  # Change this to your segmentations folder
    )
    files_info = get_files_info(folder_path)

    exclusion_file = (
        "path/to/exclusion_list.txt"  # Change this to your exclusion list file path
    )
    exclusion_list = load_exclusion_list(exclusion_file)

    for file, subject, position in files_info:
        sub_pos_key = subject + "_pos-" + position
        if sub_pos_key in exclusion_list or int(subject) <= 444 or int(subject) >= 551:
            print(
                f"Skipping excluded subject-position: {sub_pos_key} or already processed",
                flush="True",
            )
            continue
        try:
            print(f"Processing {subject} - {position}")
            raw_image_path = os.path.join(folder_path, file)

            # Load the raw image
            raw_image = sitk.ReadImage(raw_image_path)

            dimensions = raw_image.GetSize()  # Returns (x, y, z)
            axial_dim = dimensions[2]
            if axial_dim < 350 or axial_dim > 700:
                print(
                    f"Skipping {raw_image_path} because slices don't match.",
                    flush=True,
                )
                continue

            images = open_and_unzip_files(
                raw_image_path, segmentations_folder, subject, position
            )

            colon_segmentation = ColonSegmentation(
                subject,
                position,
                images.get("raw_image"),
                images.get("raw_image_spacing"),
                images.get("raw_image_origin"),
                images.get("raw_image_direction"),
                images.get("totalseg-colon.mha.gz"),
                images.get("totalseg-lung_lower_lobe_left.mha.gz"),
                images.get("totalseg-lung_lower_lobe_right.mha.gz"),
                images.get("totalseg-lung_middle_lobe_right.mha.gz"),
                images.get("totalseg-lung_upper_lobe_left.mha.gz"),
                images.get("totalseg-lung_upper_lobe_right.mha.gz"),
                images.get("totalseg-small_bowel.mha.gz"),
            )

            # Perform segmentation steps
            colon_segmentation.remove_surrounding_air()
            colon_segmentation.remove_bed()
            colon_segmentation.remove_lungs()
            colon_segmentation.segment_colon()
        except Exception as e:
            print(f"Error processing {file}: {e}", flush=True)
