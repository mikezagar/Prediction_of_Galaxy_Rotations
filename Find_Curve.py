import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from spectral_cube import SpectralCube

from Data_Collection import get_moments


outer_ring = 0

def fit_tilted_rings(file_name):

    def find_values(file_name):
        moment0, moment1 = get_moments(file_name)

        vel_map = moment1.value
        # Suppose you have:
        #   vel_map: 2D observed velocity field (mom1) with shape (ny, nx)
        #   center, inc, PA, vsys, etc. as initial guesses
        #   pixel_scale: kpc/pixel or arcsec/pixel, etc.

        #Identify valid data (mask out NaNs, low S/N, etc.)
        mask = ~np.isnan(vel_map)  # simple example
        yy, xx = np.indices(vel_map.shape)
        x_c_guess = (vel_map.shape[1] - 1)/2
        y_c_guess = (vel_map.shape[0] - 1)/2

        return xx, yy, vel_map, x_c_guess, y_c_guess, mask


    # Define a function to compute model LOS velocity for each pixel
    def tilted_ring_model(params, xx, yy, mask):
        """
        params = [x_c, y_c, vsys, inc, PA, vrot1, vrot2, vrot3, ...]
        For demonstration, let's assume a small # of rings or
        we param. vrot(r) with some function.
        """
        x_c   = params[0]
        y_c   = params[1]
        vsys  = params[2]
        inc   = params[3]          # inclination in degrees
        pa    = params[4]          # position angle in degrees
        # Next parameters define vrot as function of radius, e.g. a few ring velocities
        ring_vels = params[5:]     # e.g. one per ring if you discretize radius

        # Convert degrees -> radians
        inc_rad = np.radians(inc)
        pa_rad  = np.radians(pa)

        # radius array for each pixel
        x_rel = (xx - x_c)
        y_rel = (yy - y_c)
        # elliptical radius if we incorporate inclination
        # but let's keep it simpler: R in plane, rotate coords by PA
        # These steps vary depending on the geometry definition
        # We'll do a standard transformation:
        #   x'_rel =  x_rel * cos(pa_rad) + y_rel * sin(pa_rad)
        #   y'_rel = -x_rel * sin(pa_rad) + y_rel * cos(pa_rad)
        # then y'_rel is compressed by cos(i)
        x_prime =  x_rel*np.cos(pa_rad) + y_rel*np.sin(pa_rad)
        y_prime = -x_rel*np.sin(pa_rad) + y_rel*np.cos(pa_rad)
        y_prime_deproj = y_prime / np.cos(inc_rad)

        R = np.sqrt(x_prime**2 + y_prime_deproj**2)

        # For ring discretization, find ring index in R, e.g. each ring has a boundary
        # or param with a function, e.g. vrot(r) = ring_vels at discrete radius steps

        # We'll do a simple approach: define radial bins and assign each pixel to a ring
        # Suppose we define N_rings = len(ring_vels), radial step = R_max / N_rings
        nrings = len(ring_vels)
        R_max = np.nanmax(R[mask])
        global outer_ring
        outer_ring = R_max
        ring_edges = np.linspace(0, R_max, nrings+1)
        ring_index = np.digitize(R, ring_edges) - 1  # which ring each pixel belongs to

        # rotation speed for each pixel
        vrot_pixel = np.zeros_like(R)
        valid_ring = (ring_index >= 0) & (ring_index < nrings)
        vrot_pixel[valid_ring] = ring_vels[ring_index[valid_ring]]

        # Convert rotation speed -> LOS velocity:
        # v_model = vsys + vrot * sin(i) * cos(phi)
        # where phi = arctan2(y'_prime_deproj, x_prime)
        # but we've already aligned x_prime with major axis =>
        phi = np.arctan2(y_prime_deproj, x_prime)  # angle in plane
        v_model = vsys + vrot_pixel * np.sin(inc_rad)*np.cos(phi)

        return v_model

    # Define objective function to minimize
    def residual(params, xx, yy, vel_obs, mask):
        v_model = tilted_ring_model(params, xx, yy, mask)
        diff = (vel_obs - v_model)[mask]  # only compare valid pixels
        return diff # least_squares will minimize sum(diff^2)

    def find_curve(xx, yy, mask, vel_map, x_c_guess, y_c_guess):
        # Initial guess for parameters
        ring_count = 50
        initial_ring_vels = np.linspace(0, 200, ring_count).tolist()
        p0 = [x_c_guess, y_c_guess, 0.0, 70.0, 0.0] + initial_ring_vels

        # Fit
        res = least_squares(residual, p0, args=(xx, yy, vel_map, mask))
        best_params = res.x

        # Extract the fitted rotation curve
        best_ring_vels = best_params[5:]

        # If you want vrot vs radius, you know each ring => ring_edges

        nrings = len(best_ring_vels)
        ring_edges = np.linspace(0, outer_ring, nrings+1)
        r_mids = 0.5*(ring_edges[1:] + ring_edges[:-1])

        return r_mids, best_ring_vels

    xx, yy, vel_map, x_c_guess, y_c_guess, mask = find_values(file_name)
    r_mids, velocity = find_curve(xx, yy, mask, vel_map, x_c_guess, y_c_guess)

    return r_mids, velocity