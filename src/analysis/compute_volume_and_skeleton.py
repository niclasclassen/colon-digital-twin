import os
import csv
import argparse
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute volume and skeleton size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    optional = parser.add_argument_group("optional arguments")

    required.add_argument(
        "-i", "--input_dir",
        type=str,
        help="Define directory containing the segmentation masks in .mha format.",
        required=True,
    )

    optional.add_argument(
        "-o", "--output_path",
        type=str,
        default="segmentations_variability.csv",
        help="Define output path for .csv file.",
    )

    return parser.parse_args()

# calculate the volume of a component
def calculate_volume(component_mask, spacing):
    return np.sum(component_mask) * np.prod(spacing)


# calculate the skeleton size of a component
def calculate_centerline_length(component_mask):
    # Skeletonize the binary mask
    skeleton = skeletonize(component_mask, method="lee")
    # Count the number of skeleton pixels
    skeleton_length = np.sum(skeleton)
    return skeleton_length


def main():
    # load args
    args = parse_args()

    # Open the CSV file for writing
    with open(args.output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(["Filename", "Component", "Volume", "Centerline Length"])

        # Iterate over all .mha files in the folder
        for filename in os.listdir(args.input_dir):
            print("Processing file:", filename, flush=True)
            if filename.endswith(".mha"):
                filepath = os.path.join(args.input_dir, filename)

                # Read the .mha file
                image = sitk.ReadImage(filepath)
                image_array = sitk.GetArrayFromImage(image)
                spacing = image.GetSpacing()

                binary_mask = image_array > 0

                # Label connected components
                labeled_mask, num_features = ndimage.label(binary_mask)

                filtered_labeled_mask = np.zeros_like(labeled_mask)
                component_volumes = []
                new_component_index = 1

                for component_index in range(1, num_features + 1):
                    component_mask = labeled_mask == component_index
                    volume = calculate_volume(component_mask, spacing)

                    filtered_labeled_mask[component_mask] = new_component_index
                    centerline_length = calculate_centerline_length(
                        component_mask
                    )
                    component_volumes.append(
                        (new_component_index, volume, centerline_length)
                    )
                    new_component_index += 1

                # Write the results to the CSV
                for component_index, volume, centerline_length in component_volumes:
                    writer.writerow([filename, component_index, volume, centerline_length])

    print(f"Component analysis completed. Results saved to {args.output_path}.")

if __name__ == "__main__":
    main()
