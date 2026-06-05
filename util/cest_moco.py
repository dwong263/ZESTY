"""
CEST Motion Correction (CEST-MoCo) — Breitling et al., NMR Biomed 2022
========================================================================
Python re-implementation of MotionCorrection_PROPOSED.m from:

    Breitling et al. (2022). Motion correction for three-dimensional
    chemical exchange saturation transfer imaging without direct water
    saturation artifacts. NMR in Biomedicine, 35(7), e4720.
    https://doi.org/10.1002/nbm.4720
    https://github.com/jbreitling/CEST-MoCo

Algorithm summary
-----------------
1. **Registration**: each Z-spectrum volume is rigidly registered to a
   fixed reference volume (the M0 / unsaturated image, or a user-supplied
   index).  This gives one 4×4 rigid transform T_k per offset.

2. **AIM** (Artefact Identification and Mitigation): the transforms are
   iteratively corrected for direct-water-saturation (DWS) artefacts via
   log-matrix blending in SE(3):

   a. For each offset i, compute a Gaussian-weighted blend T̃_i of all
      transforms.  The Gaussian width σ_eff = σ / (wk_i/max(wk) + 0.2)
      where wk_i = L2-norm of the image at offset i.  DWS volumes have low
      signal → small wk_i → wide Gaussian → heavy borrowing from
      neighbours, which suppresses the artefact.

   b. Measure the RMS geodesic distance d_RMS between each T_k and T̃_k.

   c. The single worst outlier (flagged by MAD with threshold=10) is
      replaced with its T̃; repeat until no outliers remain.

3. **Resampling**: apply the (possibly corrected) transforms to resample
   each volume into the reference frame.

Dependencies
------------
    pip install nibabel SimpleITK scipy numpy

Usage
-----
    from cest_moco import run_cest_moco
    import numpy as np

    run_cest_moco(
        input_nii  = "cest_4d.nii.gz",
        output_nii = "cest_4d_moco.nii.gz",
        # Optional: index of the M0/reference volume (default: auto = brightest)
        reference_volume_index = 0,
        # Optional: supply frequency offsets to get informative logging
        offsets_ppm = np.array([-5,-4,-3,-2,-1,-0.5,0,0.5,1,2,3,4,5]),
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.linalg import expm, logm
from scipy.stats import median_abs_deviation


# ============================================================
#  Public API
# ============================================================

def run_cest_moco(
    input_nii: Union[str, Path],
    output_nii: Union[str, Path],
    *,
    reference_volume_index: Optional[int] = None,
    offsets_ppm: Optional[np.ndarray] = None,
    # AIM parameters (match MATLAB defaults exactly)
    aim_radius_mm: float = 70.0,
    aim_sigma: float = 0.6,
    aim_mad_threshold: float = 10.0,
    # Registration parameters
    registration_metric: str = "mattes_mi",
    shrink_factors: tuple = (4, 2, 1),
    smoothing_sigmas: tuple = (2.0, 1.0, 0.0),
    max_iterations: int = 100,
    verbose: bool = True,
) -> nib.Nifti1Image:
    """
    Motion-correct a 4-D CEST NIfTI image using the Breitling 2022 algorithm.

    Parameters
    ----------
    input_nii : str or Path
        4-D .nii or .nii.gz file (x, y, z, offset).
    output_nii : str or Path
        Output path for motion-corrected image.
    reference_volume_index : int, optional
        Volume index to register all others to (the 'target' in the MATLAB
        code). Defaults to the volume with the highest mean signal (M0).
    offsets_ppm : array-like, optional
        Frequency offsets in ppm — used only for verbose logging.
    aim_radius_mm : float
        Sphere radius for RMS distance calculation (default 70 mm, as in the
        paper — empirically determined for human head).
    aim_sigma : float
        Base sigma for the Gaussian blending kernel (default 0.6, paper value).
    aim_mad_threshold : float
        MAD outlier threshold factor (default 10, paper value).
    registration_metric : str
        SimpleITK metric: "mattes_mi" (default, matches MATLAB ITKRigidSlabbedHead)
        or "correlation".
    shrink_factors, smoothing_sigmas, max_iterations : registration settings.
    verbose : bool
        Print progress.

    Returns
    -------
    nib.Nifti1Image
        Motion-corrected 4-D image (also saved to output_nii).
    """
    input_nii  = Path(input_nii)
    output_nii = Path(output_nii)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if verbose:
        print(f"[CEST-MoCo] Loading {input_nii} …")
    nii_img = nib.load(str(input_nii))
    data = np.asarray(nii_img.dataobj, dtype=np.float32)

    if data.ndim != 4:
        raise ValueError(f"Expected 4-D NIfTI, got shape {data.shape}.")

    n_vols = data.shape[3]
    if verbose:
        print(f"[CEST-MoCo] {n_vols} volumes, spatial shape {data.shape[:3]}")

    # ------------------------------------------------------------------
    # 2. Choose reference volume  (= "target" in the MATLAB code)
    # ------------------------------------------------------------------
    if reference_volume_index is None:
        reference_volume_index = int(
            np.argmax([data[..., k].mean() for k in range(n_vols)])
        )
        if verbose:
            print(f"[CEST-MoCo] Auto reference volume: index {reference_volume_index} "
                  f"(mean={data[..., reference_volume_index].mean():.1f})")
    ref_idx = reference_volume_index

    # ------------------------------------------------------------------
    # 3. Build SimpleITK images (all share the same physical space)
    # ------------------------------------------------------------------
    sitk_vols = [_nifti_vol_to_sitk(data[..., k], nii_img.affine) for k in range(n_vols)]
    ref_sitk  = sitk.Cast(sitk_vols[ref_idx], sitk.sitkFloat32)

    # ------------------------------------------------------------------
    # 4. Register every volume to the reference  →  4×4 transforms T_k
    # ------------------------------------------------------------------
    if verbose:
        print(f"[CEST-MoCo] Registering {n_vols} volumes to reference …")

    T_stack = np.zeros((4, 4, n_vols))  # T_stack[:,:,k] in standard (numpy) convention
    for k in range(n_vols):
        if k == ref_idx:
            T_stack[:, :, k] = np.eye(4)
            if verbose:
                print(f"  vol {k:3d}  [reference]")
            continue

        tf = _register_rigid(sitk_vols[k], ref_sitk,
                             registration_metric, shrink_factors,
                             smoothing_sigmas, max_iterations)
        T_stack[:, :, k] = _sitk_to_matrix(tf)

        if verbose:
            t = T_stack[:3, 3, k]
            print(f"  vol {k:3d}  t=({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm")

    # ------------------------------------------------------------------
    # 5. AIM — Artefact Identification and Mitigation
    #    wk = L2-norm of each volume  (squared before passing, per MATLAB line 66)
    # ------------------------------------------------------------------
    if verbose:
        print("[CEST-MoCo] Running AIM …")

    wk_signal = np.array([float(np.linalg.norm(data[..., k])) for k in range(n_vols)])
    wk = wk_signal ** 2   # MATLAB passes wk.^2 to AIM

    T_stack = _aim(T_stack, wk,
                   R=aim_radius_mm,
                   sigma=aim_sigma,
                   mad_threshold=aim_mad_threshold,
                   verbose=verbose)

    # ------------------------------------------------------------------
    # 6. Resample each volume using the (possibly corrected) transform
    # ------------------------------------------------------------------
    if verbose:
        print("[CEST-MoCo] Resampling volumes …")

    corrected = np.zeros_like(data)
    corrected[..., ref_idx] = data[..., ref_idx]

    for k in range(n_vols):
        if k == ref_idx:
            continue
        tf_sitk = _matrix_to_sitk(T_stack[:, :, k])
        resampled = sitk.Resample(
            sitk.Cast(sitk_vols[k], sitk.sitkFloat32),
            ref_sitk,
            tf_sitk,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        # sitk → numpy: sitk is (z,y,x) after GetArrayFromImage
        corrected[..., k] = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_img = nib.Nifti1Image(corrected, nii_img.affine, nii_img.header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(output_nii))
    if verbose:
        print(f"[CEST-MoCo] Saved → {output_nii}")

    return out_img


# ============================================================
#  AIM — direct translation of the MATLAB AIM() function
# ============================================================

def _aim(
    T: np.ndarray,
    wk: np.ndarray,
    R: float = 70.0,
    sigma: float = 0.6,
    mad_threshold: float = 10.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Artefact Identification and Mitigation (AIM).

    Parameters
    ----------
    T : ndarray, shape (4, 4, n_offsets)
        Rigid 4×4 transforms in standard (numpy) convention, one per offset.
    wk : ndarray, shape (n_offsets,)
        Signal energy weight per offset (L2-norm² of the image volume).
        DWS volumes have low wk and thus receive wider Gaussian blending.
    R : float
        Radius of the region of interest in mm (default 70 mm for head).
    sigma : float
        Base Gaussian sigma for the blending kernel (default 0.6).
    mad_threshold : float
        Outlier threshold factor for MAD (default 10).

    Returns
    -------
    T : ndarray, shape (4, 4, n_offsets)
        Corrected transforms.
    """
    T = T.copy()
    n = T.shape[2]
    k_indices = np.arange(n, dtype=float)  # 0-based, mirrors MATLAB 1:N shifted by offset

    iteration = 0
    while True:
        iteration += 1
        d_rms = np.full(n, np.nan)
        T_tilde = np.zeros((4, 4, n))

        for i in range(n):
            # ----------------------------------------------------------
            # Gaussian weights  (MATLAB lines 108-109)
            # wi = wk .* exp(-((k - i)^2 / 2) / sigma_eff^2)
            # sigma_eff = sigma / (wk_i/max(wk) + 0.2)
            # ----------------------------------------------------------
            wk_norm_i = wk[i] / np.max(wk)           # normalised weight of volume i
            sigma_eff  = sigma / (wk_norm_i + 0.2)   # effective width

            wi = wk * np.exp(-((k_indices - i) ** 2) / (2.0 * sigma_eff ** 2))
            wi = wi / wi.sum()                         # normalise

            # ----------------------------------------------------------
            # Log-matrix blending in SE(3)  (MATLAB lines 112-116)
            # T̃_i = expm( Σ_k  wi_k * logm(T_k) )
            # ----------------------------------------------------------
            log_blend = np.zeros((4, 4))
            for k in range(n):
                log_blend += wi[k] * _safe_logm(T[:, :, k])

            T_tilde[:, :, i] = expm(log_blend)
            # Enforce exact bottom row (avoids tiny numerical drift)
            T_tilde[3, :, i] = [0.0, 0.0, 0.0, 1.0]

            # ----------------------------------------------------------
            # RMS geodesic distance  (MATLAB lines 120-121)
            # M = T_i * inv(T̃_i) - I
            # d_RMS = sqrt(1/5 * R² * trace(M_rot.T @ M_rot) + M_trans.T @ M_trans)
            #
            # MATLAB uses M(4,1:3) for translation because it stores
            # transforms in the transposed convention (translation in last row).
            # Here we use standard convention (translation in last column),
            # so translation deviation is M[:3, 3].
            # ----------------------------------------------------------
            M = T[:, :, i] @ np.linalg.inv(T_tilde[:, :, i]) - np.eye(4)
            M_rot   = M[:3, :3]
            M_trans = M[:3, 3]
            d_rms[i] = np.sqrt(
                (1.0 / 5.0) * R ** 2 * np.trace(M_rot.T @ M_rot)
                + M_trans @ M_trans
            )

        # --------------------------------------------------------------
        # Outlier detection via MAD  (MATLAB lines 125-131)
        # MATLAB's isoutlier(...,'ThresholdFactor',10) flags values that
        # deviate more than 10 * scaled_MAD from the median.
        # scipy's median_abs_deviation with scale='normal' applies the
        # same 1/Φ⁻¹(3/4) ≈ 1.4826 consistency factor as MATLAB.
        # --------------------------------------------------------------
        med     = np.median(d_rms)
        mad_val = median_abs_deviation(d_rms, scale="normal")
        is_outlier = np.abs(d_rms - med) > mad_threshold * mad_val

        if verbose:
            print(f"  AIM iter {iteration}: d_RMS range [{d_rms.min():.3f}, "
                  f"{d_rms.max():.3f}] mm,  outliers: {is_outlier.sum()}")

        if is_outlier.sum() == 0:
            break   # converged

        # Replace only the SINGLE worst outlier (MATLAB lines 130-131)
        worst = int(np.argmax(d_rms))
        T[:, :, worst] = T_tilde[:, :, worst]

    return T


def _safe_logm(M: np.ndarray) -> np.ndarray:
    """Matrix logarithm with fallback for near-singular matrices."""
    try:
        result = logm(M)
        # logm can return complex for near-degenerate inputs; take real part
        if np.iscomplexobj(result):
            result = result.real
        return result
    except Exception:
        return np.zeros_like(M)


# ============================================================
#  SimpleITK / NIfTI helpers
# ============================================================

def _nifti_vol_to_sitk(vol: np.ndarray, affine: np.ndarray) -> sitk.Image:
    """
    Convert a single 3-D numpy volume + NIfTI affine to a SimpleITK image
    with correct physical space metadata (origin, spacing, direction cosines).
    """
    vol_f32 = vol.astype(np.float32)

    # NIfTI stores (i,j,k) = (x,y,z); SimpleITK GetImageFromArray expects
    # the array in (z,y,x) order.
    sitk_img = sitk.GetImageFromArray(vol_f32.transpose(2, 1, 0))

    # Spacing = column norms of the affine's rotation/scale block
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).tolist()
    sitk_img.SetSpacing(spacing)

    # Origin = translation column
    sitk_img.SetOrigin(affine[:3, 3].tolist())

    # Direction = normalised columns of affine (direction cosines), flattened
    dir_mat = affine[:3, :3] / np.array(spacing)
    sitk_img.SetDirection(dir_mat.flatten().tolist())

    return sitk_img


def _register_rigid(
    moving_sitk: sitk.Image,
    fixed_sitk:  sitk.Image,
    metric: str,
    shrink_factors: tuple,
    smoothing_sigmas: tuple,
    max_iterations: int,
) -> sitk.Transform:
    """
    Rigid (Euler 3-D) registration of moving → fixed.
    Returns the final SimpleITK transform.
    """
    reg = sitk.ImageRegistrationMethod()

    if metric == "mattes_mi":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    elif metric == "correlation":
        reg.SetMetricAsCorrelation()
    elif metric == "mean_squares":
        reg.SetMetricAsMeanSquares()
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=max_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel(shrinkFactors=list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=list(smoothing_sigmas))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Geometry-based initialisation (aligns centres of mass)
    init_tf = sitk.CenteredTransformInitializer(
        fixed_sitk,
        moving_sitk,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(init_tf, inPlace=False)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_tf = reg.Execute(
                sitk.Cast(fixed_sitk,  sitk.sitkFloat32),
                sitk.Cast(moving_sitk, sitk.sitkFloat32),
            )
    except Exception as exc:
        warnings.warn(f"Registration failed ({exc}); using identity.")
        final_tf = sitk.Euler3DTransform()

    return final_tf


def _sitk_to_matrix(tf: sitk.Transform) -> np.ndarray:
    """
    Convert any SimpleITK rigid transform to a 4×4 numpy matrix in
    standard (numpy) convention: rotation block in [:3,:3], translation
    in [:3, 3], bottom row = [0,0,0,1].

    Correctly handles CompositeTransform by extracting the last component.
    """
    # Unwrap composite
    while tf.GetName() == "CompositeTransform":
        n = tf.GetNumberOfTransforms()
        tf = tf.GetNthTransform(n - 1)

    # Cast to Euler3DTransform if possible; otherwise use raw matrix/translation
    name = tf.GetName()
    if name == "Euler3DTransform":
        euler = sitk.Euler3DTransform(tf)
        A = np.array(euler.GetMatrix()).reshape(3, 3)
        t = np.array(euler.GetTranslation())
    else:
        # Generic fallback: get 3×3 matrix and translation directly
        A = np.array(tf.GetMatrix()).reshape(3, 3)
        t = np.array(tf.GetTranslation())

    T = np.eye(4)
    T[:3, :3] = A
    T[:3,  3] = t
    return T


def _matrix_to_sitk(T: np.ndarray) -> sitk.Euler3DTransform:
    """
    Convert a 4×4 numpy matrix (standard convention) to a SimpleITK
    Euler3DTransform.  Rotation is extracted via SVD to guarantee
    orthogonality after numerical operations (logm/expm can introduce drift).
    """
    R_raw = T[:3, :3]
    # Nearest orthogonal matrix via SVD
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt
    if np.linalg.det(R) < 0:   # ensure proper rotation (det=+1)
        U[:, -1] *= -1
        R = U @ Vt

    tf = sitk.Euler3DTransform()
    tf.SetMatrix(R.flatten().tolist())
    tf.SetTranslation(T[:3, 3].tolist())
    return tf


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CEST Motion Correction — Breitling 2022 (NMR Biomed e4720)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",  help="Input 4-D .nii / .nii.gz")
    parser.add_argument("output", help="Output .nii.gz")
    parser.add_argument("--ref",  type=int,   default=None,
                        help="Reference volume index (default: auto)")
    parser.add_argument("--offsets", type=float, nargs="+", default=None,
                        help="Frequency offsets in ppm (informational only)")
    parser.add_argument("--aim-radius",   type=float, default=70.0,
                        help="AIM sphere radius in mm")
    parser.add_argument("--aim-sigma",    type=float, default=0.6,
                        help="AIM Gaussian sigma")
    parser.add_argument("--aim-mad",      type=float, default=10.0,
                        help="AIM MAD threshold factor")
    parser.add_argument("--metric", default="mattes_mi",
                        choices=["mattes_mi", "correlation", "mean_squares"])
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--quiet",    action="store_true")

    args = parser.parse_args()
    run_cest_moco(
        input_nii=args.input,
        output_nii=args.output,
        reference_volume_index=args.ref,
        offsets_ppm=np.array(args.offsets) if args.offsets else None,
        aim_radius_mm=args.aim_radius,
        aim_sigma=args.aim_sigma,
        aim_mad_threshold=args.aim_mad,
        registration_metric=args.metric,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
    )