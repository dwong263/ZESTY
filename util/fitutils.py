import numpy as np
from scipy.ndimage import gaussian_filter
from lmfit import Parameters, minimize
from joblib import Parallel, delayed
import warnings

# ----------------------------
# Methods for Gaussian Fitting
# ----------------------------
def _1Gaussian(x, off, amp, cen, wid):
    return off+amp*np.exp(-0.5*(x-cen)**2/wid**2)

def _multi_gaussian(params, f):
    y = np.zeros_like(f)
    n_peaks = params['n_peaks'].value
    off     = params['offset']
    for i in range(int(n_peaks)):
        amp = params[f'amp_{i}']
        cen = params[f'cen_{i}']
        wid = params[f'wid_{i}']
        y += _1Gaussian(f, off, amp, cen, wid)
    return y

def _residual_gaussian(params, f, z_spectrum):
    return _multi_gaussian(params, f) - z_spectrum

def create_params_gaussian(n_peaks=1, p0=None):
    params = Parameters()
    params.add('n_peaks', value = n_peaks, vary=False)
    params.add('offset', value=p0[0])
    for i in range(n_peaks):
        params.add(f'amp_{i}', value=p0[i*3 + 1])
        params.add(f'cen_{i}', value=p0[i*3 + 2])
        params.add(f'wid_{i}', value=p0[i*3 + 3])
    
    return params

def fit_voxel_gaussian(z_spectrum, fdata, p0, n_peaks = 1):
    # Check if there is anything to fit in the first place
    if np.sum(z_spectrum) == 0:
        return np.full(len(p0), np.nan)

    with warnings.catch_warnings():
        params = create_params_gaussian(n_peaks, p0)
        result = minimize(
            _residual_gaussian,
            params,
            args=(fdata, z_spectrum),
            method='least_squares' # Trust Region Reflective Method (faster)
        )

    # Check fit success explicitly
    if not result.success:
        return np.full(len(p0), np.nan)
    
    # Extract fitted parameters
    fitted = [result.params['offset']]
    for i in range(n_peaks):
        fitted.extend([
            result.params[f'amp_{i}'].value,
            result.params[f'cen_{i}'].value,
            result.params[f'wid_{i}'].value,
        ])
    return np.array(fitted)

def fit_volume_parallel_gaussian(z_data, f_data, p0, mask=None, n_jobs=-1, n_peaks=1):
    nx, ny, nz, n_offsets = z_data.shape

    # Flatten spatial dimensions
    z_flat = z_data.reshape(-1, n_offsets)

    if mask is not None:
        mask_flat = mask.ravel()
        indices = np.where(mask_flat)[0]
    else:
        indices = np.arange(z_flat.shape[0])
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_voxel_gaussian)(
            z_flat[i],
            f_data,
            p0,
            n_peaks
        )
        for i in indices
    )

    # Reassemble
    n_params = len(p0)
    param_map = np.full((z_flat.shape[0], n_params), np.nan)
    param_map[indices] = results

    return param_map.reshape(nx, ny, nz, n_params)
    
# -------------------------------
# Methods for Lorentzian Fitting
# -------------------------------
def _1Lorentzian(f, A, G, d):
    return A * ((G**2/4.0)/((G**2/4.0) + (f - d)**2))

def _5Lorentzian(f, off,
                 A_water, Gamma_water, delta_water,
                 A_MT, Gamma_MT, delta_MT,
                 A_NOE, Gamma_NOE, delta_NOE,
                 A_amide_3_5, Gamma_amide_3_5, delta_amide_3_5,
                 A_amine_2_75, Gamma_amine_2_75, delta_amine_2_75):
    return off + _1Lorentzian(f, A_water, Gamma_water, delta_water) \
               + _1Lorentzian(f, A_MT, Gamma_MT, delta_MT) \
               + _1Lorentzian(f, A_NOE, Gamma_NOE, delta_NOE) \
               + _1Lorentzian(f, A_amide_3_5, Gamma_amide_3_5, delta_amide_3_5) \
               + _1Lorentzian(f, A_amine_2_75, Gamma_amine_2_75, delta_amine_2_75)

def _7Lorentzian(f, off,
                 A_water, Gamma_water, delta_water,
                 A_MT, Gamma_MT, delta_MT,
                 A_NOE, Gamma_NOE, delta_NOE,
                 A_NOE2, Gamma_NOE2, delta_NOE2,
                 A_amide_3_5, Gamma_amide_3_5, delta_amide_3_5,
                 A_amine_2_75, Gamma_amine_2_75, delta_amine_2_75,
                 A_amine_2, Gamma_amine_2, delta_amine_2):
    return off + _1Lorentzian(f, A_water, Gamma_water, delta_water) \
               + _1Lorentzian(f, A_MT, Gamma_MT, delta_MT) \
               + _1Lorentzian(f, A_NOE, Gamma_NOE, delta_NOE) \
               + _1Lorentzian(f, A_NOE2, Gamma_NOE2, delta_NOE2) \
               + _1Lorentzian(f, A_amide_3_5, Gamma_amide_3_5, delta_amide_3_5) \
               + _1Lorentzian(f, A_amine_2_75, Gamma_amine_2_75, delta_amine_2_75) \
               + _1Lorentzian(f, A_amine_2, Gamma_amine_2, delta_amine_2)

def _multi_lorentzian(params, f):
    y = np.zeros_like(f)
    n_peaks = params['n_peaks'].value
    for i in range(int(n_peaks)):
        A = params[f'amp_{i}']
        G = params[f'width_{i}']
        d = params[f'shift_{i}']
        y += _1Lorentzian(f, A, G, d)
    return y + params['offset']

def _residual_lorentzian(params, f, z_spectrum):
    return _multi_lorentzian(params, f) - z_spectrum

def create_params_lorentzian(p0, bounds):
    params = Parameters()

    n_peaks = int((len(p0)-1)/3)
    params.add('n_peaks', value=n_peaks, vary=False)
    
    params.add('offset', value=p0[0], min=bounds[0][0], max=bounds[0][1])
    for i in range(n_peaks):
        params.add(f'amp_{i}',   value=p0[i*3 + 1], min=bounds[i*3 + 1][0], max=bounds[i*3 + 1][1])
        params.add(f'width_{i}', value=p0[i*3 + 2], min=bounds[i*3 + 2][0], max=bounds[i*3 + 2][1])
        params.add(f'shift_{i}', value=p0[i*3 + 3], min=bounds[i*3 + 3][0], max=bounds[i*3 + 3][1])

    return params

def fit_voxel_lorentzian(z_spectrum, fdata, p0, bounds,
                         n_retries=5, perturb_scale=0.1,
                         seed=None, redchi_threshold=0.01, 
                         rel_residual_threshold=0.05):
    
    # Check if there is anything to fit in the first place
    if np.sum(z_spectrum) == 0:
        return np.full(len(p0), np.nan)

    # Detect the number of peaks
    n_peaks = int((len(p0)-1)/3)

    # Setup Fit
    p0_arr = np.array(p0)   # initial guess
    rng = np.random.default_rng(seed)   # random range for perturbation of initial guess
    lo = np.array([b[0] for b in bounds])   # lo limit for perturbation of initial guess
    hi = np.array([b[1] for b in bounds])   # hi limit for perturbation of initial guess

    # Method to Check Fit Success
    def _is_acceptable(result):
        # Primary: optimizer converged
        if result.success:
            return True
        # Secondary: fit is good even if optimizer flagged non-convergence
        redchi_ok = (result.redchi is not None) and (result.redchi < redchi_threshold)
        residuals = result.residual
        rel_residual = np.sqrt(np.mean(residuals**2)) / (np.ptp(z_spectrum) + 1e-12)
        rel_residual_ok = rel_residual < rel_residual_threshold
        return redchi_ok and rel_residual_ok

    # Method to Extract Result
    def _extract(result):
        fitted = [result.params['offset'].value]
        for i in range(n_peaks):
            fitted.extend([
                result.params[f'amp_{i}'].value,
                result.params[f'width_{i}'].value,
                result.params[f'shift_{i}'].value,
            ])
        return np.array(fitted)

    # Method to Fit
    def _try_fit(p0_attempt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            params = create_params_lorentzian(p0_attempt.tolist(), bounds)
            result = minimize(
                _residual_lorentzian,
                params,
                args=(fdata, z_spectrum),
                method='least_squares' # Trust Region Reflective Method (faster)
            )
            return result
    
    # Try Fits Until Convergence
    for attempt in range(n_retries+1):
        if attempt == 0:
            p0_attempt = p0_arr.copy()
        else:
            noise = rng.normal(0, perturb_scale * (hi - lo), size=len(p0_arr))
            p0_attempt = np.clip(p0_arr + noise, lo, hi)

        result = _try_fit(p0_attempt)

        if _is_acceptable(result):
            return _extract(result)

    # Return NaN If No Convergence Within n_retries     
    return np.full(len(p0), np.nan)

def fit_volume_parallel_lorentzian(z_data, f_data, p0, bounds, 
                                   mask=None, n_jobs=-1,
                                   n_retries=5, perturb_scale=0.1):
    
    nx, ny, nz, n_offsets = z_data.shape

    # Flatten spatial dimensions
    z_flat = z_data.reshape(-1, n_offsets)
    f_flat = f_data.reshape(-1, n_offsets)

    if mask is not None:
        mask_flat = mask.ravel()
        indices = np.where(mask_flat)[0]
    else:
        indices = np.arange(z_flat.shape[0])

    # Parallel execution
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_voxel_lorentzian)(
            z_flat[i],
            f_flat[i],
            p0,
            bounds,
            n_retries=n_retries,
            perturb_scale=perturb_scale,
            seed=i  # unique but reproducible seed per voxel      
        )
        for i in indices
    )

    # Reassemble
    n_params = len(p0)
    param_map = np.full((z_flat.shape[0], n_params), np.nan)
    param_map[indices] = results

    return param_map.reshape(nx, ny, nz, n_params)

# --------------------------------
# Methods for Smoothing AACID Maps
# --------------------------------
def gaussian_smooth(img, sigma):
    if sigma == 0:
        return img
    else:
        return gaussian_filter(img, sigma=sigma)

def gaussian_smooth_ignore_nan(img, sigma):
    if sigma == 0:
        return img

    # Step 1: define mask (valid = 1, missing = 0)
    mask = (img != 0).astype(float)

    # Step 2: replace missing values with 0 (so they don't contribute)
    img_filled = np.where(mask, img, 0)

    # Step 3: smooth both image and mask
    smoothed_img = gaussian_filter(img_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(mask, sigma=sigma)

    # Step 4: normalize
    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed_img / smoothed_mask

    # Optional: reassign missing voxels
    result[smoothed_mask == 0] = 0

    return result