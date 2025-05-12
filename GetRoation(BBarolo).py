import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from Data_Collection import get_moments

file_path = '/Users/mikezagar/Desktop/ENPH 455/Training Data/losvd_FIRE_i2_100_gas_incl70.fits'

data = fits.open(file_path)
X = data[0].data

mom0, mom1 = get_moments(file_path)

from mpl_toolkits.axes_grid1 import make_axes_locatable
vel_curve_file = '/Users/mikezagar/Desktop/ENPH 455/FIRE_incl70/NONE_2drings.txt'
radius, vsys, vrot, pa, incl  = np.loadtxt(vel_curve_file, usecols = (1, 2, 3, 5, 6), unpack=True)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(3, 2, figsize =(14, 16))
fig.suptitle(r'Data obtained from WALLABY of J103915-301757', y = 1)

im1 = ax[0,0].imshow(mom0.data)
ax[0,0].set_ylabel(r'$DEC^\circ$')
ax[0,0].set_xlabel(r'$RA^\circ$')
ax[0,0].set_title('Moment 0 Map')
ax[0,0].invert_yaxis()

divider1 = make_axes_locatable(ax[0,0])
cax1 = divider1.append_axes("right", size="5%", pad=0.1)  # Adjust size and pad as needed
cbar1 = plt.colorbar(im1, cax=cax1, label = r'$\frac{Hz*Jy}{Beam}$')

im2 = ax[0,1].imshow(mom1.data,vmin=1.401e9,vmax=1.404e9, cmap = 'bwr')
ax[0,1].set_title('Moment 1 Map')
ax[0,1].set_ylabel(r'$DEC^\circ$')
ax[0,1].set_xlabel(r'$RA^\circ$')
ax[0,1].invert_yaxis()

divider2 = make_axes_locatable(ax[0,1])
cax2 = divider2.append_axes("right", size="5%", pad=0.1)  # Adjust size and pad as needed
cbar2 = plt.colorbar(im2, cax=cax2, label = r'$Hz$')

ax[1,0].plot(radius, vrot, marker = 'o', label="Velocity Curve", color = 'blue')
ax[1,1].plot(radius, vsys, marker = 'o', color = 'blue')
ax[2,0].plot(radius, pa, marker = 'o', color = 'blue')
ax[2,1].plot(radius, incl, marker = 'o', color = 'blue')

ax[1,0].set_title('Velocty Curve')
ax[1,1].set_title('Total Velocity')
ax[2,0].set_title('Position Angle')
ax[2,1].set_title('Inclination')

ax[1,0].set_ylabel(r'$V_{rot}[/frac{km}{s}]$')
ax[1,1].set_ylabel(r'$V_{sys}[\frac{km}{s}]$')
ax[2,0].set_ylabel(r'$PA^\circ$')
ax[2,1].set_ylabel(r'$Incl^\circ$')

ax[1,0].set_xlabel(r'Radius[arcs]')
ax[1,1].set_xlabel(r'Radius[arcs]')
ax[2,0].set_xlabel(r'Radius[arcs]')
ax[2,1].set_xlabel(r'Radius[arcs]')

fig.tight_layout()
plt.show()