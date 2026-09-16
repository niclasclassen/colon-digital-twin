import os
import csv
import argparse
import heapq

import numpy as np
import SimpleITK as sitk

from scipy.ndimage import label
from skimage.morphology import skeletonize

# GLOBAL VARIABLES
HEADER_ADDED = False


def component_geodesic_diameter(component_mask, spacing_xyz):
    """
    Approximate the main geodesic path length of one connected
    3D skeleton component using two Dijkstra sweeps.

    component_mask: bool array with shape (z, y, x)
    spacing_xyz: (sx, sy, sz) in mm

    Returns:
        diameter_mm
    """

    coords = np.argwhere(component_mask)

    if len(coords) <= 1:
        return 0.0

    coord_set = {tuple(c) for c in coords}

    sx, sy, sz = spacing_xyz
    offsets = []

    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue

                edge_length = np.sqrt((dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2)
                offsets.append((dz, dy, dx, edge_length))

    def dijkstra(start):
        distances = {start: 0.0}
        queue = [(0.0, start)]

        farthest_node = start
        farthest_distance = 0.0

        while queue:
            current_distance, node = heapq.heappop(queue)

            if current_distance != distances[node]:
                continue

            if current_distance > farthest_distance:
                farthest_distance = current_distance
                farthest_node = node

            z, y, x = node

            for dz, dy, dx, weight in offsets:
                neighbor = (z + dz, y + dy, x + dx)

                if neighbor not in coord_set:
                    continue

                new_distance = current_distance + weight

                if new_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_distance
                    heapq.heappush(queue, (new_distance, neighbor))

        return farthest_node, farthest_distance

    start = tuple(coords[0])

    endpoint_a, _ = dijkstra(start)
    endpoint_b, diameter_mm = dijkstra(endpoint_a)

    return diameter_mm


def compute_componentwise_skeleton_length(skeleton, spacing_xyz):
    """
    Finds one main geodesic path per connected skeleton component
    and sums those path lengths.

    Returns:
        total_length_mm
        component_lengths_mm
    """

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_skeleton, n_components = label(skeleton, structure=structure)

    component_lengths_mm = []

    for component_id in range(1, n_components + 1):
        component = labeled_skeleton == component_id

        length_mm = component_geodesic_diameter(component, spacing_xyz)

        component_lengths_mm.append(length_mm)

    total_length_mm = sum(component_lengths_mm)

    return (total_length_mm, component_lengths_mm)


def remove_small_components(mask, spacing_xyz, min_volume_ml=5.0):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_mask, n_components = label(mask, structure=structure)

    component_sizes = np.bincount(labeled_mask.ravel())[1:]

    voxel_volume_ml = np.prod(spacing_xyz) / 1000.0
    min_voxels = int(np.ceil(min_volume_ml / voxel_volume_ml))

    cleaned_mask = np.zeros_like(mask, dtype=bool)

    for component_id, size in enumerate(component_sizes, start=1):
        if size >= min_voxels:
            cleaned_mask[labeled_mask == component_id] = True

    return cleaned_mask


def analyze_colon(mask_path):
    """
    Compute colon segmentation metrics from an MHA file.
    """

    image = sitk.ReadImage(mask_path)
    spacing = image.GetSpacing()
    array = sitk.GetArrayFromImage(image)

    mask = array > 0
    mask = remove_small_components(mask, spacing, min_volume_ml=5.0)

    # Basic volume information
    n_voxels = int(np.count_nonzero(mask))
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = n_voxels * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0

    # Connected components of colon
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_mask, n_components = label(mask, structure=structure)
    component_sizes = np.bincount(labeled_mask.ravel())[1:]

    if len(component_sizes) > 0:
        largest_component_voxels = int(component_sizes.max())
        largest_component_fraction = largest_component_voxels / n_voxels
    else:
        largest_component_voxels = 0
        largest_component_fraction = 0.0

    # 3D skeleton
    skeleton = skeletonize(mask, method="lee")
    n_skeleton_voxels = int(np.count_nonzero(skeleton))

    # Approximate main-path length
    skeleton_length_mm, component_lengths_mm = compute_componentwise_skeleton_length(
        skeleton, spacing
    )
    skeleton_length_cm = skeleton_length_mm / 10.0
    component_lengths_cm = [length_mm / 10.0 for length_mm in component_lengths_mm]

    metrics = {
        "file": os.path.basename(mask_path),
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "colon_voxels": n_voxels,
        "volume_ml": volume_ml,
        "skeleton_length_cm": skeleton_length_cm,
        "connected_components": int(n_components),
        "largest_component_fraction": largest_component_fraction,
    }

    return metrics

def save_data(path: str, metrics: dict):
    global HEADER_ADDED
    if not HEADER_ADDED:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        HEADER_ADDED = True
    else:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writerow(metrics)

def main():
    parser = argparse.ArgumentParser(
        description=("Compute variability metrics from a colon MHA segmentation.")
    )
    parser.add_argument("--input_mha", help=("Path to the colon segmentation .mha file"))
    parser.add_argument("--output", default="colon_metrics_daria.csv", help="Output CSV path")
    args = parser.parse_args()

    for filename in sorted(os.listdir(args.input_mha)):
            print("Processing file:", filename, flush=True)
            if filename.endswith(".mha"):
                filepath = os.path.join(args.input_mha, filename)
                metrics = analyze_colon(filepath)

                save_data(args.output, metrics)                

                print("\nColon metrics\n")

                for key, value in metrics.items():
                    if isinstance(value, (float, np.floating)):
                        print(f"{key}: {value:.4f}")

                    else:
                        print(f"{key}: {value}")

                print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
