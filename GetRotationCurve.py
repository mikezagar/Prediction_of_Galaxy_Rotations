import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from astropy.io import fits
from astropy.visualization import simple_norm

from Data_Collection import get_moments

from Find_Curve import *
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm

file_path = '/Users/mikezagar/Desktop/ENPH 455/Training Image Data/losvd_FIRE_i2_100_gas_incl70.fits'

data = fits.open(file_path)
X = data[0].data

mom0, mom1 = get_moments(file_path)

r_mids, velocity = fit_tilted_rings(file_path)

smoothed_curve = savgol_filter(velocity, 7, 3)
smoothed_gauss = gaussian_filter1d(velocity, sigma=2)

lowess = sm.nonparametric.lowess(velocity, r_mids, frac=0.2)
radius_loess, smoothed_loess = lowess[:,0], lowess[:,1]

plt.figure(figsize = (10,6))
plt.rcParams["font.family"] = "Times New Roman"

# Moment-0: integrated intensity
'''plt.subplot(221)
norm0 = simple_norm(mom0.value, stretch='linear', percent=99)
plt.imshow(mom0.value, origin='lower', cmap='inferno', norm=norm0)
plt.colorbar(label='Integrated Intensity [unit × km/s]')
plt.title('Moment 0 Map')

plt.subplot(222)
norm1 = simple_norm(mom1.value, stretch='linear', percent=99)
plt.imshow(mom1.value, origin='lower', cmap='coolwarm', norm=norm1)
plt.colorbar(label='Mean Velocity [km/s]')
plt.title('Moment 1 Map')'''

plt.subplot(121)
plt.plot(r_mids, velocity, 'o-', color='red', markersize=5, linewidth=1)
plt.title('Derived Rotation Curve')
plt.ylabel('V [km/s]')
plt.xlabel('R ["]')

plt.subplot(122)
plt.plot(r_mids, smoothed_gauss, '--', color='blue', label="Gaussian Smoothing")
plt.plot(r_mids, smoothed_curve, '-.', color='green', label="Svaitzky-Golay Smoothing")
plt.plot(radius_loess, smoothed_loess, '--', color='red', label="LOESS Smoothing")
plt.title('Smoothed Rotation Curves')
plt.ylabel('V [km/s]')
plt.xlabel('R ["]')
plt.legend()
plt.tight_layout()

plt.savefig('/Users/mikezagar/Desktop/ENPH 455/Images/Smoothed Rotation Curves.png')
plt.show()
