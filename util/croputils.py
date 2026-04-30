import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

def _get_world_corners(img):
    """Return 8 world-space corners of a NIfTI image."""
    shape = img.shape[:3]
    affine = img.affine

    ijk = np.array([
        [0, 0, 0],
        [shape[0], 0, 0],
        [0, shape[1], 0],
        [0, 0, shape[2]],
        [shape[0], shape[1], 0],
        [shape[0], 0, shape[2]],
        [0, shape[1], shape[2]],
        [shape[0], shape[1], shape[2]],
    ])

    ijk_h = np.c_[ijk, np.ones(8)]
    xyz = (affine @ ijk_h.T).T[:, :3]
    return xyz


def _world_box_to_voxel_bounds(img, world_min, world_max):
    """
    Convert world overlap box to voxel slicing bounds (half-open).
    Fully orientation-safe.
    """
    affine_inv = np.linalg.inv(img.affine)

    # Construct full 8-corner overlap cube in world space
    x0, y0, z0 = world_min
    x1, y1, z1 = world_max

    world_corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y1, z0],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],
    ])

    wc_h = np.c_[world_corners, np.ones(8)]
    vox = (affine_inv @ wc_h.T).T[:, :3]

    vmin = np.ceil(vox.min(axis=0)).astype(int)
    vmax = np.floor(vox.max(axis=0)).astype(int)

    # Clip to valid half-open bounds
    shape = np.array(img.shape[:3])
    vmin = np.maximum(vmin, 0)
    vmax = np.minimum(vmax, shape)

    return vmin, vmax


def _crop_nifti(img, vmin, vmax):
    """Crop NIfTI image safely, preserving dtype and header."""
    data = img.dataobj  # avoids float64 inflation

    slicer = (
        slice(vmin[0], vmax[0]),
        slice(vmin[1], vmax[1]),
        slice(vmin[2], vmax[2])
    )

    if img.ndim == 4:
        slicer = slicer + (slice(None),)

    cropped = np.asanyarray(data[slicer])

    new_affine = img.affine.copy()
    new_affine[:3, 3] = (
        img.affine @ np.array([vmin[0], vmin[1], vmin[2], 1])
    )[:3]

    new_header = img.header.copy()
    new_header.set_data_shape(cropped.shape)

    return nib.Nifti1Image(cropped, new_affine, new_header)


def crop_to_common_overlap(
    img1,
    img2,
    resample=False,
    resample_order=3
):
    """
    Robust cropping to common spatial overlap.

    Parameters
    ----------
    img1, img2 : nib.Nifti1Image
    resample : bool
        If True, resample img2 to img1 grid before cropping.
    resample_order : int
        Interpolation order if resampling (0=nearest, 1=linear, 3=cubic).

    Returns
    -------
    img1_crop, img2_crop
        Cropped NIfTI images containing identical world-space region.
    """

    if resample:
        img2 = resample_from_to(img2, img1, order=resample_order)

    # World-space bounding boxes via 8-corner method
    corners1 = _get_world_corners(img1)
    corners2 = _get_world_corners(img2)

    min1, max1 = corners1.min(axis=0), corners1.max(axis=0)
    min2, max2 = corners2.min(axis=0), corners2.max(axis=0)

    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)

    if np.any(overlap_max <= overlap_min):
        raise ValueError("Images do not overlap in world space.")

    # Convert overlap box back to voxel bounds
    vmin1, vmax1 = _world_box_to_voxel_bounds(img1, overlap_min, overlap_max)
    vmin2, vmax2 = _world_box_to_voxel_bounds(img2, overlap_min, overlap_max)

    img1_crop = _crop_nifti(img1, vmin1, vmax1)
    img2_crop = _crop_nifti(img2, vmin2, vmax2)

    return img1_crop, img2_crop