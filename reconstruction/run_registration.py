import SimpleITK as sitk
import os
import re

# --- Configuration ---
fixed_image_path = "path/to/your/fixed_image"  # Replace with the actual path to your fixed image (image with collapsed colon)
moving_images_dirs = [
    "moving_inages_dir_1/",  # Replace with actual paths to directories containing moving images (non-collapsed) (in our case, images were split into two directories)
    "moving_inages_dir_2/",
]
segmentation_dir = "segmentation_masks_dir/"  # Replace with the actual path to the directory containing segmentation masks

# Load fixed image
fixed_ct = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)


# helper function to compute mutual information
def compute_mi(fixed, moving):
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)
    registration.SetInterpolator(sitk.sitkLinear)
    initial_tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial_tx, inPlace=False)
    return -registration.MetricEvaluate(fixed, moving)


# Search for best match
best_mi = -float("inf")
best_image_path = None
best_seg_path = None

for moving_dir in moving_images_dirs:
    try:
        for file in os.listdir(moving_dir):
            if not file.endswith(".mha") or "_0000.mha" not in file:
                continue

            match = re.match(r"colon_0*(\d+)-(supine|prone)_0000\.mha", file)
            if not match:
                print(f"Filename doesn't match expected pattern: {file}", flush=True)
                continue

            sub_id = match.group(1).zfill(3)
            position = match.group(2)  # "supine" or "prone"

            seg_file = f"sub{sub_id}_pos-{position}_scan-1.mha"
            seg_path = os.path.join(segmentation_dir, seg_file)
            image_path = os.path.join(moving_dir, file)

            if not os.path.exists(seg_path):
                print(f"Segmentation not found: {seg_file}", flush=True)
                continue

            try:
                moving_ct_tmp = sitk.ReadImage(image_path, sitk.sitkFloat32)
                mi = compute_mi(fixed_ct, moving_ct_tmp)
                print(f"{file} -> MI: {mi:.4f}", flush=True)

                if mi > best_mi:
                    best_mi = mi
                    best_image_path = image_path
                    best_seg_path = seg_path
            except Exception as e:
                print(f"Skipping {file} due to read/similarity error: {e}", flush=True)
    except FileNotFoundError:
        print(f"Directory not found: {moving_dir}", flush=True)


# Result
if best_image_path:
    print(
        f"\nBest match:\nImage: {best_image_path}\nSeg: {best_seg_path}\nMI: {best_mi:.4f}",
        flush=True,
    )
else:
    print("\nNo valid moving image with matching segmentation found.", flush=True)

# --- Run registration with best image ---
moving_ct = sitk.ReadImage(best_image_path, sitk.sitkFloat32)
non_collapsed_seg = sitk.ReadImage(best_seg_path, sitk.sitkFloat32)

# Step 3: Initial transform
initial_translation = sitk.CenteredTransformInitializer(
    fixed_ct,
    moving_ct,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)

# Step 4–5: B-spline transform
mesh_size = [2] * fixed_ct.GetDimension()
bspline_transform = sitk.BSplineTransformInitializer(fixed_ct, mesh_size)
composite_transform = sitk.CompositeTransform(3)
composite_transform.AddTransform(initial_translation)
composite_transform.AddTransform(bspline_transform)

# Step 6: Registration setup
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetInitialTransform(composite_transform, inPlace=False)
registration_method.SetMetricAsMattesMutualInformation(50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.1)
registration_method.SetInterpolator(sitk.sitkLinear)
registration_method.SetOptimizerAsLBFGSB(
    gradientConvergenceTolerance=1e-5,
    numberOfIterations=100,
    maximumNumberOfCorrections=5,
    maximumNumberOfFunctionEvaluations=1000,
    costFunctionConvergenceFactor=1e7,
)
registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Step 7: Run registration
registration_method.AddCommand(
    sitk.sitkIterationEvent,
    lambda: print(
        f"Iter: {registration_method.GetOptimizerIteration()}, "
        f"Metric: {registration_method.GetMetricValue():.6f}",
        flush=True,
    ),
)
final_transform = registration_method.Execute(fixed_ct, moving_ct)

# Step 8: Resample segmentation
transformed_seg = sitk.Resample(
    non_collapsed_seg,
    fixed_ct,
    final_transform,
    sitk.sitkNearestNeighbor,
    0.0,
    sitk.sitkFloat32,
)

# Step 9: Save result
sitk.WriteImage(transformed_seg, "transformed_seg.mha")  # Adjust output path as needed
