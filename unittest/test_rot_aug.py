from feature_robustness_analysis.rot_aug.im_ops import *
from feature_robustness_analysis.rot_aug.rot_ops import *
import unittest
import numpy as np
import SimpleITK as sitk

class TestRotAugImgOps(unittest.TestCase):
    def setUp(self) -> None:
        # Create a binary label image with three connected bodies of different sizes
        self.dummy_label = sitk.Image([100, 100, 10], sitk.sitkUInt8)
        self.dummy_label[2 : 50, 2 : 50, 2: 5]  = 1
        self.dummy_label[51: 95, 51: 95, 5: 8]  = 2
        self.dummy_label[1 : 40, 7 : 90, 8: 10] = 3

    def test_list_of_connected_bodies_sorted_by_size(self):
        r"""Tests that the list of connected bodies is sorted by size, from largest to smallest."""
        label = self.dummy_label

        # Call the function
        bodies = get_connected_bodies(label)

        # Check that the output is a list of three images
        self.assertIsInstance(bodies, list)
        self.assertEqual(3, len(bodies))

        # Check that the images are sorted by size, from largest to smallest
        sizes = [np.count_nonzero(sitk.GetArrayFromImage(body)) for body in bodies]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_smallest_bounding_box_with_padding_0(self):
        label = sitk.Image((10, 10, 10), sitk.sitkUInt8)
        label[3, 3, 3] = 1
        expected_output = (3, 3, 3, 3, 3, 3)
        self.assertEqual(expected_output, get_bounding_box(label, padding=0))

    def test_smallest_bounding_box_with_padding_5(self):
        label = sitk.Image((20, 20, 20), sitk.sitkUInt8)
        label[3, 3, 3] = 1
        expected_output = (0, 0, 0, 13, 13, 13)
        self.assertEqual(expected_output, get_bounding_box(label, padding=5))

    def test_get_cent_of_mass(self):
        r"""Test that the function returns the correct center of mass
        for a label image containing a very large object
        """
        label = sitk.Image((100, 100, 100), sitk.sitkUInt8)
        label[40:60, 40:60, 40:60] = 1
        expected_com = (49.5, 49.5, 49.5)
        result = get_cent_of_mass(label)
        self.assertEqual(result, expected_com)

    def test_resample_image_valid_input_affine_transform(self):
        """Tests that resample_image function works correctly with valid input and affine transform"""
        # Create a larger image
        image = sitk.Image([20, 20, 20], sitk.sitkUInt8)
        image.SetSpacing([1, 1, 1])
        image.SetOrigin([0, 0, 0])
        image.SetDirection(np.identity(3).flatten())
        # Create an affine transform
        affine_transform = rot_to_affine([R(10, 20)], [(10, 10, 10)])[0]
        # Resample the image
        resampled_image = resample_image(image, affine_transform)
        # Check that the output image has the correct size, spacing, origin and direction
        self.assertEqual(resampled_image.GetSize(), (20, 20, 20))
        self.assertTrue(np.allclose(resampled_image.GetSpacing(), [1, 1, 1]))
        self.assertTrue(np.allclose(resampled_image.GetOrigin(), [0, 0, 0]))
        self.assertTrue(np.allclose(resampled_image.GetDirection(), np.identity(3).flatten()))

    def test_resample_image_valid_input_affine_transform_w_bbox(self):
        """Tests that resample_image function works correctly with valid input and affine transform"""
        # Create a larger image
        image = sitk.Image([20, 20, 20], sitk.sitkUInt8)
        image.SetSpacing([1, 1, 1])
        image.SetOrigin([0, 0, 0])
        image.SetDirection(np.identity(3).flatten())
        # Create an affine transform
        affine_transform = rot_to_affine([R(10, 20)], [(10, 10, 10)])[0]
        # Resample the image
        resampled_image = resample_image(image, affine_transform)
        # Check that the output image has the correct size, spacing, origin and direction
        self.assertEqual(resampled_image.GetSize(), (20, 20, 20))
        self.assertTrue(np.allclose(resampled_image.GetSpacing(), [1, 1, 1]))
        self.assertTrue(np.allclose(resampled_image.GetOrigin(), [0, 0, 0]))
        self.assertTrue(np.allclose(resampled_image.GetDirection(), np.identity(3).flatten()))

    def test_resample_to_segment_bbox(self):
        r"""Tests that the function correctly resamples when img and seg have the same spacing, size, and direction"""
        img = sitk.Image([100, 100, 10], sitk.sitkUInt8)
        seg = sitk.Image([100, 100, 10], sitk.sitkUInt8)
        resampled_img, resampled_seg = resample_to_segment_bbox(img, seg)
        self.assertTrue(np.allclose(resampled_img.GetSize(), resampled_seg.GetSize()))
        self.assertTrue(np.allclose(resampled_img.GetSpacing(), resampled_seg.GetSpacing()))
        self.assertTrue(np.allclose(resampled_img.GetDirection(), resampled_seg.GetDirection()))

    def test_simple_input_vector_rotation(self):
        in_vec = np.array([1, 0, 0])
        rot = transform.Rotation.from_euler('x', 45, degrees=True)
        self.assertEqual(45, calculate_rotated_angle(in_vec, rot))