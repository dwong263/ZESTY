import numpy as np
from lmfit import Parameters, minimize
from joblib import Parallel, delayed
import warnings

# -------------------------
# Methods for Gaussian Fitting
# -------------------------
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
    
# -------------------------
# Methods for Lorentzian Fitting
# -------------------------
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

def create_params_lorentzian(n_peaks=7, p0=None):
    params = Parameters()
    params.add('n_peaks', value=n_peaks, vary=False)

    BOUNDS_5L = [
        [0.5, 1.0],                                 # vertical offset
        [-1.0, -0.02], [0.3, 10.0],  [-1.0, 1.0],   # water
        [-0.5, 0.0],   [30.0, 60.0], [-2.5, 0.0],   # MT
        [-0.2, 0.0],   [0.4, 6.0],   [-4.0, -3.0],  # NOE (-3.5 ppm)
        [-0.2, 0.0],   [0.4, 6.0],   [3.2, 3.8],    # amide (3.5 ppm)
        [-0.1, 0.0],   [0.4, 6.0],   [2.5, 3.0]     # amine (2.75 ppm)
    ]
    
    BOUNDS_7L = [
        [0.5, 1.0],                                 # vertical offset
        [-1.00, -0.02], [0.30, 10.0], [-1.0, 1.0],  # water
        [-0.25, -0.02], [7.00, 25.0], [-3.0, 0.0],  # MT
        [-0.15, -0.02], [2.00, 15.0], [-3.2, -2.8], # NOE    (-3.0 ppm)
        [-0.15, -0.02], [1.00, 15.0], [-4.2, -3.8], # NOE2   (-4.0 ppm)
        [-0.20, 0.00],  [0.40, 6.00], [3.4, 3.6],   # amide  (3.5 ppm)
        [-0.20, 0.00],  [0.40, 6.00], [2.5, 2.8],   # amine  (2.75 ppm)
        [-0.20, 0.00],  [0.40, 6.00], [1.9, 2.1]    # amine2 (2.0 ppm)
    ]

    if n_peaks == 5:
        params.add('offset', value=p0[0], min=BOUNDS_5L[0][0], max=BOUNDS_5L[0][1])
        for i in range(n_peaks):
            params.add(f'amp_{i}',   value=p0[i*3 + 1], min=BOUNDS_5L[i*3 + 1][0], max=BOUNDS_5L[i*3 + 1][1])
            params.add(f'width_{i}', value=p0[i*3 + 2], min=BOUNDS_5L[i*3 + 2][0], max=BOUNDS_5L[i*3 + 2][1])
            params.add(f'shift_{i}', value=p0[i*3 + 3], min=BOUNDS_5L[i*3 + 3][0], max=BOUNDS_5L[i*3 + 3][1])
    elif n_peaks == 7:
        params.add('offset', value=p0[0], min=BOUNDS_7L[0][0], max=BOUNDS_7L[0][1])
        for i in range(n_peaks):
            params.add(f'amp_{i}',   value=p0[i*3 + 1], min=BOUNDS_7L[i*3 + 1][0], max=BOUNDS_7L[i*3 + 1][1])
            params.add(f'width_{i}', value=p0[i*3 + 2], min=BOUNDS_7L[i*3 + 2][0], max=BOUNDS_7L[i*3 + 2][1])
            params.add(f'shift_{i}', value=p0[i*3 + 3], min=BOUNDS_7L[i*3 + 3][0], max=BOUNDS_7L[i*3 + 3][1])

    return params

def fit_voxel_lorentzian(z_spectrum, fdata, p0, n_peaks=7):
    # Fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        params = create_params_lorentzian(n_peaks, p0)
        result = minimize(
            _residual_lorentzian,
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
            result.params[f'width_{i}'].value,
            result.params[f'shift_{i}'].value,
        ])
    return np.array(fitted)

def fit_volume_parallel_lorentzian(z_data, f_data, p0, mask=None, n_jobs=-1, model='5L'):
    
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
    if model == '5L':
        n_peaks = 5
    elif model == '7L':
        n_peaks = 7
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_voxel_lorentzian)(
            z_flat[i],
            f_flat[i],
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