import os
import SimpleITK as sitk
import numpy as np
import argparse

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

    required.add_argument(
        "-o", "--output_dir",
        type=str,
        help="Define output directory",
        required=True,
    )

    optional.add_argument(
        "--min_cmp_size",
        type=int,
        default=500,
        help="Define min component size to be included.",
        required=True,
    )

    return parser.parse_args()

def main():
    # load args
    args = parse_args()

    # make sure target directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop through all .mha files in the folder
    for file_name in sorted(os.listdir(args.input_dir)):
        if not file_name.endswith(".mha"):
            continue

        input_path = os.path.join(args.input_dir, file_name)
        output_path = os.path.join(args.output_dir, file_name)

        image = sitk.ReadImage(input_path)
        array = sitk.GetArrayFromImage(image)

        # remove all connected components smaller than args.min_cmp_size voxels
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_image = cc_filter.Execute(image)
        relabel_filter = sitk.RelabelComponentImageFilter()
        relabel_filter.SetMinimumObjectSize(args.min_cmp_size)
        cleaned_image = relabel_filter.Execute(cc_image)

        total = cc_filter.GetObjectCount()
        remaining = relabel_filter.GetNumberOfObjects()
        print(
            f"Removed {total - remaining} of {total} connected components (< {args.min_cmp_size} voxels): {file_name}"
        )

        # set all nonzero values to 1 again after relabeling
        cleaned_array = sitk.GetArrayFromImage(cleaned_image)
        cleaned_array = (cleaned_array > 0).astype(np.uint8)
        cleaned_image_binarized = sitk.GetImageFromArray(cleaned_array)
        cleaned_image_binarized.CopyInformation(image)

        # Save result
        sitk.WriteImage(cleaned_image_binarized, output_path)
        print(f"Binarized: {file_name}", flush=True)

    print("All masks processed.")

if __name__ == "__main__":
    main()
