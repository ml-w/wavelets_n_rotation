"""
Filename: rot_ops.py
Author: lwong
Date Created: 2023/8/9
Date Last Modified: 2023/8/9
Description: This file hold code for generating rotation matrices and convertion between rotation and affine matrix.
"""
import numpy as np
import SimpleITK as sitk
from scipy.spatial import transform
from typing import Iterable, Optional, Tuple, Union, List, Dict

# Get Euler angle rotational matrix. intrinsic rotation. yaw then pitch
R = lambda yaw, pitch: transform.Rotation.from_euler('ZX', [yaw, pitch], degrees=True)


def calculate_rotated_angle(in_vec: np.ndarray, rot: transform.Rotation):
    """Calculate the angle between the rotated vector and the original vector"""
    rot_vec = rot.apply(in_vec)
    cos_theta = np.linalg.norm(np.cross(in_vec, rot_vec)) / np.linalg.norm(in_vec) / np.linalg.norm(rot_vec)
    theta = np.rad2deg(np.arcsin(cos_theta)) # convert to degree
    return theta


def gen_rot_set(size: int,
                pitch_mu: float,
                pitch_sigma: float,
                yaw_mu: float,
                yaw_sigma: float) -> Iterable[transform.Rotation]:
    r"""Generates a set of rotation matrices based on normally distributed pitch and yaw angles.

    This function creates a specific number of rotation matrices by generating a set of pitch and yaw angles.
    The pitch angles are normally distributed with a user-specified mean and standard deviation. The yaw angles
    are normally distributed with a mean of 0 and a standard deviation of 180.

    Args:
        size (int):
            The number of rotation matrices to generate.
        pitch_mu (float):
            The mean of the pitch angles in degrees.
        pitch_sigma (float):
            The standard deviation of the pitch angles in degrees.
        yaw_mu (float):
            The mean of the yaw angles in degrees.
        yaw_sigma (float):
            The standadr deviation of the yaw angles in degrees.

    Returns:
        Iterable[transform.Rotation]:
            A list of rotation matrices, each represented as a `transform.Rotation` object.

    Raises:
        ValueError:
            If `size` is less than 1, or `pitch_sigma` is less than 0.
    """
    # sample pitch and yaw
    PHI = np.random.normal(size=size, scale=pitch_sigma, loc=pitch_mu)
    YAW = np.random.normal(size=size, scale=yaw_sigma, loc=yaw_mu)

    # rotation matrices
    out = [R(yaw, phi) for yaw, phi in zip(YAW, PHI)]
    return out


def rot_to_affine(rotations: Iterable[transform.Rotation],
                  centers: Iterable[Tuple[float, float, float]]) -> List[sitk.AffineTransform]:
    r"""Convert a set of scipy Rotation objects into a list of SimpleITK AffineTransform objects
    about a specified center.

    Args:
        rotations (Iterable[Rotation]): A list of rotation objects from scipy.
        center (Tuple[float, float, float]): The center about which the rotations should be performed.
                                              Specified as a tuple of three coordinates (x, y, z).

    Returns:
        List[sitk.AffineTransform]: A list of sitk.AffineTransform objects, each representing one of the input rotations.
    """
    # make sure the number of rotation and the rotation center is the same
    if not len(rotations) == len(centers):
        msg = f"Length of rotations ({len(rotations)}) is not the same as the centers ({len(centers)})."
        raise IndexError(msg)

    # make sure the rotation centers are specified as (x, y, z)
    if not all([len(c) == 3 for c in centers]):
        msg = f"Some of the centers provided are not 3-vectors."
        raise ValueError(msg)

    affines = []
    for rot, cent in zip(rotations, centers):
        affine = sitk.AffineTransform(3)
        affine.SetCenter(cent)
        matrix = rot.as_matrix()
        affine.SetMatrix(matrix.ravel())
        affines.append(affine)
    return affines


def gen_afftine_mat(pitch_mu: float,
                    pitch_sigma: float,
                    yaw_mu: float,
                    yaw_sigma: float,
                    size: int,
                    rot_cent: Iterable[Tuple[float, float, float]],
                    keys: Iterable[str]) -> Dict[str, Iterable[sitk.AffineTransform]]:
    r"""Generate an affine matrix from a rotation

    Args:
        pitch_mu (float):
            The mean of the pitch angles in degrees.
        pitch_sigma (float):
            The standard deviation of the pitch angles in degrees.
        yaw_mu (float):
            The mean of the yaw angles in degrees.
        yaw_sigma (float):
            The standard deviation of the yaw angles in degrees.
        size (int):
            The number of rotation matrices to generate.
        rot_cent (Iterable[Tuple[float, float, float]]):
            The center of the rotation.
        keys (Iterable[str]):
            The keys that pairs with `rot_cent`.

    Returns:
        dict:
            A dictionary of rotation matrices, each represented as a `transform.Rotation` object.
    """
    Rset = gen_rot_set(size, pitch_mu=pitch_mu, pitch_sigma=pitch_sigma, yaw_mu=yaw_mu, yaw_sigma=yaw_sigma)
    Rset = rot_to_affine(Rset, [np.array(v[0]) for v in rot_cent])
    Rset = {key: r for key, r in zip(keys, Rset)}
    return Rset