import numpy as np
import math
from scipy.optimize import curve_fit

from Data_Augmentation import dataCube
from spectral_cube import SpectralCube
from astropy.io import fits

def get_moments(file_path):
    hdu = dataCube(file_path)
    hdu.writeto("fixed_cube_for_spectralcube.fits", overwrite=True)

    cube = SpectralCube.read("fixed_cube_for_spectralcube.fits")
    moment0 = cube.moment(order=0, axis=0)
    moment1 = cube.moment(order=1, axis=0)

    return moment0, moment1

def get_rotation(file_path, moment1):
    hdul = fits.open(file_path, cache=True, show_progress=True)
    data = hdul[0].data
    header = hdul[0].header
    hdul.close()

    mom1_data = moment1.value  # shape => (ny, nx)
    ny, nx = mom1_data.shape

    # 4) Define pixel->kpc scale and center
    pixel_scale_kpc = header.get('FOVSIZE')  # from your header, e.g. PIXSIZE
    xc = (nx - 1) / 2.0
    yc = (ny - 1) / 2.0

    Y, X = np.indices((ny, nx))
    r_pix = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    r_kpc = r_pix * pixel_scale_kpc

    # 5) Bin radius
    r_max = np.nanmax(r_kpc)
    nbins = 20
    rbins = np.linspace(0, r_max, nbins + 1)
    r_mids = 0.5 * (rbins[:-1] + rbins[1:])

    vel_curve_obs = np.zeros(nbins)
    for i in range(nbins):
        mask = (r_kpc >= rbins[i]) & (r_kpc < rbins[i + 1])
        vel_curve_obs[i] = np.nanmean(mom1_data[mask])

    # 6) Correct for inclination (optional)
    incl_deg = header.get('INCL')  # from your header
    i_rad = math.radians(incl_deg)
    vel_curve_rot = vel_curve_obs / np.sin(i_rad)

    return r_mids, vel_curve_obs

def arctan_model(r, v0, vmax, rt):
    return v0 + (2.0/np.pi)*vmax*np.arctan(r/rt)

def extrapolate_arctan(v0, vmax, rt):
    r_mid = np.linspace(0.5, 10, 10)  # dummy radius bins
    # Let's create some synthetic "data" using known params + noise
    true_params = (v0, vmax, rt)  # (v0, vmax, rt)
    rng = np.random.default_rng(42)
    vel_curve_clean = arctan_model(r_mid, *true_params)
    vel_curve = vel_curve_clean + rng.normal(0, 10, size=r_mid.size)

    p0 = [50.0, 150.0, 1.0]
    params, cov = curve_fit(arctan_model, r_mid, vel_curve, p0=p0)

    v0, vmax, rt = params
    errors = np.sqrt(np.diag(cov))  # standard deviations
    err_v0, err_vmax, err_rt = errors

    print("Fitted parameters:")
    print(f"  v0    = {v0:.3f} ± {err_v0:.3f} km/s")
    print(f"  vmax  = {vmax:.3f} ± {err_vmax:.3f} km/s")
    print(f"  r_t   = {rt:.3f} ± {err_rt:.3f} kpc")

    r_fit = np.linspace(0, r_mid.max()*1.1, 200)
    vel_model = arctan_model(r_fit, *params)

    return r_mid, r_fit, vel_model, vel_curve

