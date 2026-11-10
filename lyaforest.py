"""
lyaforest: synthetic Lyman-alpha absorption spectra along lines of sight.

Given the HI number density, temperature and peculiar velocity along a
skewer, compute_tau returns the Lyman-alpha optical depth in each pixel by
summing Voigt-profile contributions from every pixel along the skewer
(e.g. Choudhury, Srianand & Padmanabhan 2001):

    tau(z_0) = (c I_alpha dx / sqrt(pi)) * sum_x n_HI(x) H(a, u) / [b(x) (1 + z(x))]

with u = [v_pec(x) + H(z_0) dx_sep / (1 + z_0)] / b(x) and
a = Gamma lambda_alpha / (4 pi b(x)), where dx_sep is the comoving separation
from pixel x to the pixel at z_0, wrapped around the box if it is periodic.

Usage
-----
    import lyaforest

    # Any skewer: arrays of shape (N_pixel,) or (N_pixel, N_los)
    tau = lyaforest.compute_tau(n_HI, T, v_pec, z, L_box=40)

    # Skewers from the tutorial simulation output
    skewer = lyaforest.read_skewer(redshift_key='2.0', skewer_indx=0)
    tau = lyaforest.compute_tau(skewer['n_HI'], skewer['T'], skewer['v_pec'], skewer['z'], L_box=skewer['L_box'])
"""

import os
from functools import lru_cache

import numpy as np
from scipy.special import wofz

# Physical constants (cgs)
MASS_PROTON = 1.6726231e-24 # g
MASS_ELECTRON = 9.1093897e-28 # g
LIGHT_SPEED = 2.99792458e10 # cm/s
BOLTZMANN_CONSTANT = 1.380658e-16 # ergs/K

# Lyman-alpha transition of HI
LAMBDA_REST = 1215.6701 # Transition wavelength in Angstroms
F_12 = 0.416 # Oscillator strength for transition
DAMP_GAMMA = 6.265e8 # Damping constant of transition in s^-1
I_ALPHA = 4.45e-18 # Cross section in cm^2
MASS_ABSORBER = MASS_PROTON + MASS_ELECTRON # Mass of an HI atom in g

# Unit conversions
KMS_TO_CMS = 1e5 # km/s to cm/s
MPC_TO_CM = 3.086e24 # Mpc to cm
ANGSTROM_TO_CM = 1e-8 # Angstrom to cm

def compute_voigt(pixel_indx, z_val, z_arr, v_arr_1d, b_arr_1d, n_arr_1d, lambda_rest_cm, damp_gamma, dx, hubble_z, periodic=True):
    """Sum of n_HI * Voigt profile / [b (1+z)] over the skewer, for the pixel at z_val (all inputs cgs)."""

    N_pixel = len(z_arr)

    # Signed pixel offset from pixel_indx; for a periodic box, wrapped into [-N/2, N/2)
    pixel_offset = np.arange(N_pixel) - pixel_indx
    if periodic:
        pixel_offset = (pixel_offset + N_pixel // 2) % N_pixel - N_pixel // 2

    # Hubble velocity of that offset: H(z) * proper separation (dx is comoving)
    v_hubble_arr = hubble_z * pixel_offset * dx / (1 + z_val)

    # Compute velocity factor needed for Voigt profile calculation
    velocity_factor_arr = (v_arr_1d + v_hubble_arr) / b_arr_1d

    # Compute alpha factor needed for Voigt profile calculation
    alpha_factor_arr = (damp_gamma * lambda_rest_cm) / (4.0 * np.pi * b_arr_1d)

    # Construct complex variable for Faddeeva function
    z_complex_arr = velocity_factor_arr + 1j * alpha_factor_arr

    # Taking the real part of the Faddeeva function gives us the Voigt profile
    voigt_calc = wofz(z_complex_arr).real

    # Integrating over the line of sight to get the optical depth value for the pixel at z_val
    tau_val_pixel = np.sum(n_arr_1d * voigt_calc/(b_arr_1d * (1 + z_arr)), axis=0)

    return tau_val_pixel

def compute_tau(n_HI, T, v_pec, z, L_box, hubble_param=0.678, omega_m=0.308, omega_l=0.692, periodic=True, axis=0):
    """Lyman-alpha optical depth along one or more skewers from any simulation.

    n_HI         : HI number density (proper, cm^-3), one skewer (N_pixel,) or several (N_pixel, N_los)
    T            : gas temperature (K), same shape as n_HI
    v_pec        : peculiar velocity along the line of sight (km/s), same shape as n_HI
    z            : redshift of each pixel, shape (N_pixel,), or one redshift for the whole skewer
    L_box        : comoving length of the skewer (cMpc/h); pixels are equally spaced along it
    hubble_param, omega_m, omega_l : cosmology (defaults: that of the example simulation)
    periodic     : wrap absorption around the ends of the skewer (True for skewers through a periodic box)
    axis         : axis of n_HI, T, v_pec along the skewer (use axis=1 or -1 for (N_los, N_pixel) arrays)

    Returns the optical depth, with the same shape as n_HI.
    """

    if not (np.shape(n_HI) == np.shape(T) == np.shape(v_pec)):
        raise ValueError(f"n_HI, T and v_pec must have the same shape; got {np.shape(n_HI)}, {np.shape(T)}, {np.shape(v_pec)}")

    # Work with (N_pixel, N_los) arrays
    n_HI = np.moveaxis(np.asarray(n_HI, dtype=float), axis, 0)
    out_shape = n_HI.shape
    N_pixel = out_shape[0]

    n_arr = n_HI.reshape(N_pixel, -1)
    T_arr = np.moveaxis(np.asarray(T, dtype=float), axis, 0).reshape(N_pixel, -1)
    v_arr = np.moveaxis(np.asarray(v_pec, dtype=float), axis, 0).reshape(N_pixel, -1) * KMS_TO_CMS # in cm/s
    z_arr = np.broadcast_to(np.asarray(z, dtype=float), (N_pixel,))
    N_los = n_arr.shape[1]

    # Thermal Doppler parameter b = sqrt(2 k T / m)
    b_arr = np.sqrt(2.0 * BOLTZMANN_CONSTANT * T_arr / MASS_ABSORBER)

    lambda_rest_cm = LAMBDA_REST * ANGSTROM_TO_CM
    L_box_cm = L_box * MPC_TO_CM / hubble_param # in ccm
    dx = L_box_cm / N_pixel
    H0 = 100.0 * hubble_param * KMS_TO_CMS / MPC_TO_CM # in s^-1

    tau_arr = np.zeros((N_pixel, N_los))

    for los_indx in range(N_los):
        for pixel_indx in range(N_pixel):
            z_val = z_arr[pixel_indx]
            hubble_z = H0 * np.sqrt(omega_m * (1 + z_val)**3 + omega_l)
            tau_val = compute_voigt(pixel_indx, z_val, z_arr, v_arr[:,los_indx], b_arr[:,los_indx], n_arr[:,los_indx],
                                    lambda_rest_cm, DAMP_GAMMA, dx, hubble_z, periodic)
            tau_arr[pixel_indx, los_indx] = tau_val * LIGHT_SPEED * I_ALPHA * dx / np.sqrt(np.pi)

    return np.moveaxis(tau_arr.reshape(out_shape), 0, axis)

# ---------------------------------------------------------------------------
# Reader for the simulation output used in the tutorial
# ---------------------------------------------------------------------------

# Small example dataset shipped with the repository (4 snapshots x 10 skewers).
EXAMPLE_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_data', 'spectra_fields_example.npz')

@lru_cache(maxsize=1)
def load_spectra_fields(data_file):
    """Load the simulation output as {redshift_key: {field: array}}.

    Reads either the full pickled Spectra_Fields.npy or the example .npz file
    (keys '<redshift>/<field>'). Cached, so repeated reads do not reload the file.
    """

    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if data_file.endswith('.npz'):
        spectra_fields = {}
        with np.load(data_file) as data:
            for key in data.files:
                redshift_key, field = key.split('/')
                spectra_fields.setdefault(redshift_key, {})[field] = data[key]
        return spectra_fields

    return np.load(data_file, allow_pickle=True, encoding='latin1').item()

def read_skewer(data_file=EXAMPLE_DATA_FILE, redshift_key='2.0', skewer_indx=0):
    """Read skewers from the tutorial simulation output.

    data_file    : the example data (default) or the full Spectra_Fields.npy
    redshift_key : snapshot; '2.0', '3.0', '4.0', '5.0' in the example data,
                   '2.0' to '6.0' in steps of 0.1 in the full data
    skewer_indx  : an index (0 to 9 in the example data, 0 to 499 in the full data),
                   or a list/slice of indices to get arrays of shape (N_pixel, N_los)

    Returns a dict with n_HI (cm^-3), T (K), v_pec (km/s), z, the simulation's
    own optical depth tau_sim, and the box size L_box (cMpc/h).
    """

    spectra_fields = load_spectra_fields(data_file)
    if redshift_key not in spectra_fields:
        raise KeyError(f"Snapshot '{redshift_key}' not in {data_file}; "
                       f"available: {sorted(spectra_fields, key=float)}")
    snapshot = spectra_fields[redshift_key]

    return {
        'n_HI': snapshot['n_HI_arr'][:, skewer_indx],
        'T': snapshot['T_gas_arr'][:, skewer_indx],
        'v_pec': snapshot['v_gas_arr'][:, skewer_indx],
        'z': snapshot['z_arr'],
        'tau_sim': snapshot['tau_HI_arr'][:, skewer_indx],
        'L_box': 40.0, # Box size of the tutorial simulation in cMpc/h (not stored in the file)
    }

if __name__ == '__main__':

    skewer = read_skewer()
    tau = compute_tau(skewer['n_HI'], skewer['T'], skewer['v_pec'], skewer['z'], L_box=skewer['L_box'])

    flux = np.exp(-tau)
    flux_sim = np.exp(-skewer['tau_sim'])

    print("Example data: snapshot z = 2.0, skewer 0")
    print(f"Mean flux (computed)   : {flux.mean():.4f}")
    print(f"Mean flux (simulation) : {flux_sim.mean():.4f}")
    print(f"Max |flux difference|  : {np.abs(flux - flux_sim).max():.4f}")
