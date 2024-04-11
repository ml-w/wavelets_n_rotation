"""
Filename: im_ops.py
Author: lwong
Date Created: 2023/8/9
Date Last Modified: 2023/8/9
Description: This file hold code for image operations
"""

import SimpleITK as sitk
import numpy as np
from typing import Iterable, Optional, Tuple, Union, Callable


def get_connected_bodies(label: sitk.Image) -> Iterable[sitk.Image]:
    r"""Calculates, filters and sorts connected bodies in a label image.

    This function identifies the connected bodies in the provided label image, filters bodies
    that contain fewer than 30 pixels, and returns the remaining bodies in a list ordered by
    their size.

    Args:
        label (sitk.Image): The input binary label image where objects of interest are
        represented by non-zero pixels and the background by zero pixels.

    Returns:
        List[sitk.Image]: A list of label images. Each image corresponds to a single connected
        body from the original label image. The bodies are ordered by size (number of pixels),
        from largest to smallest, and bodies with fewer than 30 pixels are excluded.
    """
    # Label the connected components in the image
    confil = sitk.ConnectedComponentImageFilter()
    labeled = confil.Execute(label)

    # Get the number of labels
    num_labels = confil.GetObjectCount()

    # Store the separated label images in a list
    body_list = []

    # Iterate over each label
    for i in range(1, num_labels + 1):
        # Threshold the labeled image to get a binary image representing the current body
        body = sitk.BinaryThreshold(labeled, lowerThreshold=i, upperThreshold=i)

        # Convert the image to a NumPy array
        body_array = sitk.GetArrayFromImage(body)

        # Count the number of non-zero pixels (i.e., the size of the body)
        body_size = np.count_nonzero(body_array)
        num_slices = np.asarray([np.count_nonzero(np.any(body_array, axis=axis)) for axis in [(1, 2), (0, 1), (0, 2)]])

        # Filter out bodies with fewer than 30 pixels
        if body_size >= 30 and all(num_slices > 1):
            body_list.append((body_size, body))

    # Sort the bodies by size, from largest to smallest
    body_list.sort(reverse=True, key=lambda x: x[0])

    # Return only the bodies, not their sizes
    return [body for _, body in body_list]

def get_bounding_box(label: sitk.Image, padding: Optional[int] = 0) -> Tuple[int, int, int, int, int, int]:
    r"""Return the smallest bounding box of the binary label padded by the `padding` argument.

    Args:
        label (sitk.Image): The label for which the bounding box is to be calculated.
        padding (Optional[int], optional): The padding to add to the bounding box.
                                           Defaults to 0.

    Returns:
        Tuple[int, int, int, int, int, int]: The smallest bounding box of the input label,
                                              represented as a tuple of six integers.
                                              The first three integers are the starting
                                              indices in each dimension (x, y, z),
                                              and the last three integers are the sizes
                                              in each dimension.
    """
    # Calculate the bounding box
    label_statistics_filter = sitk.LabelStatisticsImageFilter()
    label_statistics_filter.Execute(label, label)
    bounding_box = list(label_statistics_filter.GetBoundingBox(1))  # 1 is the label value

    # Add padding
    for i in range(3):
        bounding_box[i] = max(0, bounding_box[i] - padding)
        bounding_box[i+3] = min(bounding_box[i+3] + 2 * padding, label.GetSize()[i] - bounding_box[i])

    return tuple(bounding_box)

def get_cent_of_mass(label: sitk.Image) -> Tuple[float, float, float]:
    r"""Computes the center of mass of an input SimpleITK Image object.

    This function calculates the center of mass in 3D space from a SimpleITK Image object.
    The input image is assumed to be a label image where the object of interest is represented
    by non-zero pixel values. The center of mass is computed in physical (not index) coordinates.

    Args:
        label (sitk.Image):
            A SimpleITK Image object. It should be a label image where the object
            of interest is represented by non-zero pixel values.

    Returns:
        Tuple[float, float, float]:
            The center of mass of the image object. The return value is a tuple
            containing the x, y, and z coordinates of the center of mass.

    Raises:
        TypeError:
            If the input label is not a SimpleITK Image object.
        ValueError:
            If the input image is not 3D.
        ValueError:
            If the input image does not contain any non-zero pixels.
    """
    if not isinstance(label, sitk.Image):
        raise TypeError("The input label must be a SimpleITK Image object.")

    if not label.GetPixelID() == sitk.sitkUInt8:
        msg = f"Expect type of image to be sitkUInt8, got {label.GetPixelIDTypeAsString()} instead."

    if label.GetDimension() != 3:
        raise ValueError("The input image must be 3D.")

    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(label, label)

    labels = stats.GetLabels()
    if not labels:
        raise ValueError("The input image does not contain any non-zero pixels.")

    return stats.GetCenterOfGravity(labels[0])


def resample_image(image: sitk.Image, affine_transform: sitk.AffineTransform = None) -> sitk.Image:
    """Resample the input image using the given affine transform. The resampled space is configured
    such that the image direction remains unchanged, but the origin is shifted such that the rotation
    center the `affine_transform` specified is the new center of the resampled space. In addition,
    the spacing is redefined to be isometric, taking the value from the finest axis.

    Args:
        image (sitk.Image): The input image to be resampled.
        affine_transform (sitk.AffineTransform): The affine transform to be applied to the input image.

    Returns:
        sitk.Image: The resampled image.
    """
    try:
        # Calculate the required shift
        spacing = np.asarray(image.GetSpacing())
        new_spacing = np.asarray([min(spacing)] * 3)
        size = np.asarray(image.GetSize())
        new_size = np.round(size * spacing / new_spacing)

        # Get the rotation center from affine matrix
        if affine_transform is not None:
            rot_cent = np.asarray(affine_transform.GetCenter())
            new_origin = rot_cent - 0.5 * new_size * new_spacing
        else:
            # If not provided, just use the original origin
            new_origin = image.GetOrigin()


        # Set default value to be minimal of the input array
        min_val = float(sitk.GetArrayFromImage(image).min())

        # Transform the image
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(new_origin)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(tuple(new_size.astype('int').tolist()))
        if not affine_transform is None:
            resampler.SetTransform(affine_transform)
        resampler.SetDefaultPixelValue(min_val)

        # Interpolation strategy changes if its a label
        if image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkLabelUInt8):
            resampler.SetInterpolator(sitk.sitkLabelGaussian)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)


        # conduct resampling
        resampled_image = resampler.Execute(image)
        return resampled_image
    except Exception as e:
        print('Error occurred during resampling:', e)
        raise e

def resample_to_segment_bbox(img: sitk.Image, seg: sitk.Image, padding: Optional[int] = 0) -> Tuple[sitk.Image, sitk.Image]:
    r"""Resample the input image and segmentation to the bounding box of the segmentation with optional padding. Padding is
    done with the minimal value of the input and default to no padding. Checking is done to ensure that `img` and `seg` has the
    same spacing, size and direction.

    Args:
        img (sitk.Image):
            The input image to be resampled.
        seg (sitk.Image):
            The segmentation image whose bounding box is used for resampling.
        padding (Optional[int], optional):
            The padding to add to the bounding box. Defaults to 0.

    Returns:
        Tuple[sitk.Image, sitk.Image]: The resampled input image and segmentation.
    """
    # Check that img and seg have the same spacing, size, and direction
    if not all([img.GetSpacing() == seg.GetSpacing(),
               img.GetSize() == seg.GetSize(),
               img.GetDirection() == seg.GetDirection()]):
        # Resample if there's differences
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        seg = resampler.Execute(seg)
    if len(np.unique(sitk.GetArrayFromImage(seg))) == 2:
        temp = sitk.GetArrayFromImage(seg)
        counts = {k: np.sum(temp == k) for k in np.unique(temp) if k != 0}
        msg = f"Input segmentation is not binary. {counts}"

    # Calculate the bounding box of the segmentation
    label_statistics_filter = sitk.LabelStatisticsImageFilter()
    label_statistics_filter.Execute(seg, seg)
    bounding_box = list(label_statistics_filter.GetRegion(1))  # 1 is the label value

    # Add padding
    for i in range(3):
        bounding_box[i] = max(0, bounding_box[i] - padding)
        bounding_box[i+3] = min(bounding_box[i+3] + 2 * padding, seg.GetSize()[i] - bounding_box[i])

    # Create a region of interest filter for resampling
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(bounding_box[:3])
    roi_filter.SetSize(bounding_box[3:])

    # Resample the images
    resampled_img = roi_filter.Execute(img)
    resampled_seg = roi_filter.Execute(seg)
    return resampled_img, resampled_seg

