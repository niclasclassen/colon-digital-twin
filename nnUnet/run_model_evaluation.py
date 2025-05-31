import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops
import torch
from monai.metrics import (
    HausdorffDistanceMetric,
    compute_average_surface_distance,
    compute_dice,
)

# Define paths
predictions_folder = (
    "predictions_path"  # Replace with the path to your predictions folder
)
ground_truth_folder = (
    "ground_truth_path"  # Replace with the path to your ground truth folder
)
output_csv_path = "results.csv"  # Replace with the desired output CSV file path

# Define the Hausdorff Distance Metric with 95th percentile
hd95_metric = HausdorffDistanceMetric(percentile=95, distance_metric="euclidean")


# Function to compute Hausdorff distance (95th percentile) using MONAI
def compute_hausdorff_distance_monai(pred_component, ground_truth):
    pred_array = sitk.GetArrayFromImage(pred_component).astype(bool)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth).astype(bool)

    if not np.any(pred_array) or not np.any(ground_truth_array):
        return np.inf  # Return infinity if one of the arrays is empty

    # Convert to PyTorch tensors and add batch and channel dimensions
    pred_tensor = torch.tensor(pred_array[None, None, ...], dtype=torch.float32)
    ground_truth_tensor = torch.tensor(
        ground_truth_array[None, None, ...], dtype=torch.float32
    )

    # Compute Hausdorff distance (95th percentile)
    hd95_value = hd95_metric(pred_tensor, ground_truth_tensor)
    return hd95_value.item()  # Convert to a scalar


# Function to compute Average Symmetric Surface Distance (ASSD)
def compute_assd(pred_component, ground_truth):
    pred_array = sitk.GetArrayFromImage(pred_component).astype(bool)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth).astype(bool)

    if not np.any(pred_array) or not np.any(ground_truth_array):
        return np.inf  # Return infinity if one of the arrays is empty

    # Convert to PyTorch tensors and add batch and channel dimensions
    pred_tensor = torch.tensor(pred_array[None, None, ...], dtype=torch.float32)
    ground_truth_tensor = torch.tensor(
        ground_truth_array[None, None, ...], dtype=torch.float32
    )

    # Compute ASSD
    assd = compute_average_surface_distance(
        y_pred=pred_tensor, y=ground_truth_tensor, symmetric=True
    )
    return assd.mean().item()  # Return the mean symmetric surface distance


# Function to compute Dice Score
def compute_dice_score(pred_component, ground_truth):
    pred_array = sitk.GetArrayFromImage(pred_component).astype(bool)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth).astype(bool)

    if not np.any(pred_array) or not np.any(ground_truth_array):
        return 0.0  # Return 0 if one of the arrays is empty

    # Convert to PyTorch tensors and add batch and channel dimensions
    pred_tensor = torch.tensor(pred_array[None, None, ...], dtype=torch.float32)
    ground_truth_tensor = torch.tensor(
        ground_truth_array[None, None, ...], dtype=torch.float32
    )

    # Compute Dice Score
    dice_score = compute_dice(
        y_pred=pred_tensor, y=ground_truth_tensor, include_background=False
    )
    return dice_score.mean().item()  # Return the mean Dice Score


# Function to compute percentage overlap
def compute_percentage_overlap(pred_component, ground_truth):
    pred_array = sitk.GetArrayFromImage(pred_component)
    ground_truth_array = sitk.GetArrayFromImage(ground_truth)

    intersection = np.logical_and(pred_array, ground_truth_array).sum()
    pred_volume = pred_array.sum()

    if pred_volume == 0:
        return 0.0  # Avoid division by zero

    return (intersection / pred_volume) * 100


# Initialize results list
results = []

# Iterate over prediction files
for pred_filename in os.listdir(predictions_folder):
    if pred_filename.endswith(".mha"):
        # Load prediction and ground truth
        pred_path = os.path.join(predictions_folder, pred_filename)
        ground_truth_path = os.path.join(ground_truth_folder, pred_filename)

        if not os.path.exists(ground_truth_path):
            print(f"Ground truth for {pred_filename} not found. Skipping.")
            continue

        pred_image = sitk.ReadImage(pred_path)
        ground_truth_image = sitk.ReadImage(ground_truth_path)

        # Compute overall Hausdorff distance for the entire prediction
        overall_hausdorff_distance = compute_hausdorff_distance_monai(
            pred_image, ground_truth_image
        )
        overall_assd = compute_assd(pred_image, ground_truth_image)

        # Label disconnected components in the prediction
        pred_array = sitk.GetArrayFromImage(pred_image)
        labeled_pred_array = label(pred_array)
        labeled_pred_image = sitk.GetImageFromArray(labeled_pred_array)
        labeled_pred_image.CopyInformation(pred_image)

        # Iterate over components in the prediction
        for region in regionprops(labeled_pred_array):
            component_id = region.label

            # Extract the current component
            component_mask = (labeled_pred_array == component_id).astype(np.uint8)
            component_image = sitk.GetImageFromArray(component_mask)
            component_image.CopyInformation(pred_image)

            # Compute metrics
            hausdorff_distance = compute_hausdorff_distance_monai(
                component_image, ground_truth_image
            )
            overlap_percentage = compute_percentage_overlap(
                component_image, ground_truth_image
            )
            assd = compute_assd(component_image, ground_truth_image)
            dice_score = compute_dice_score(component_image, ground_truth_image)
            overall_dice_score = compute_dice_score(pred_image, ground_truth_image)
            component_size = component_mask.sum()

            # Append results
            results.append(
                {
                    "filename": pred_filename,
                    "component_id": component_id,
                    "hausdorff_distance_95th": hausdorff_distance,
                    "overlap_percentage": overlap_percentage,
                    "component_size": component_size,
                    "overall_hausdorff_distance_95th": overall_hausdorff_distance,
                    "average_symmetric_surface_distance": assd,
                    "overall_average_symmetric_surface_distance": overall_assd,
                    "dice_score": dice_score,
                    "overall_dice_score": overall_dice_score,
                }
            )

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
