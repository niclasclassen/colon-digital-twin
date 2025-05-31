import os
import csv
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

# Define the folder containing the .mha files
input_folder = "path_to_segmentation_masks"  # Replace with your input folder path
output_csv = "results.csv"  # Output CSV file to save the results

# Define a threshold for filtering small components (e.g., minimum volume)
min_volume_threshold = 100


# Function to calculate the volume of a component
def calculate_volume(component_mask, spacing):
    return np.sum(component_mask) * np.prod(spacing)


# Function to calculate the skeleton size of a component
def calculate_centerline_length(component_mask, spacing):
    # Skeletonize the binary mask
    skeleton = skeletonize(component_mask, method="lee")
    # Count the number of skeleton pixels
    skeleton_length = np.sum(skeleton)
    return skeleton_length


# Open the CSV file for writing
with open(output_csv, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(["Filename", "Component", "Volume", "Centerline Length"])

    # Iterate through all .mha files in the folder
    for filename in os.listdir(input_folder):
        print("Processing file:", filename, flush=True)
        if filename.endswith(".mha"):
            filepath = os.path.join(input_folder, filename)

            # Read the .mha file
            image = sitk.ReadImage(filepath)
            image_array = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()

            binary_mask = image_array > 0

            # Label connected components
            labeled_mask, num_features = ndimage.label(binary_mask)

            # Filter out small components
            filtered_labeled_mask = np.zeros_like(labeled_mask)
            component_volumes = []
            new_component_index = 1

            for component_index in range(1, num_features + 1):
                component_mask = labeled_mask == component_index
                volume = calculate_volume(component_mask, spacing)
                if volume >= min_volume_threshold:
                    filtered_labeled_mask[component_mask] = new_component_index
                    centerline_length = calculate_centerline_length(
                        component_mask, spacing
                    )
                    component_volumes.append(
                        (new_component_index, volume, centerline_length)
                    )
                    new_component_index += 1

            # Write the results to the CSV
            for component_index, volume, centerline_length in component_volumes:
                writer.writerow([filename, component_index, volume, centerline_length])

print(f"Component analysis completed. Results saved to {output_csv}.")
