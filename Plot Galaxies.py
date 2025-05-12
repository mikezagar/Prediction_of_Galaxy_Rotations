import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

file_path = '/Users/mikezagar/Desktop/ENPH 455/Training Data/losvd_FIRE_i2_100_gas_incl70.fits'
hdu = fits.open(file_path)

X = hdu[0].data

plt.rcParams["font.family"] = "Times New Roman"
plt.imshow(X[100,:,:], cmap='inferno')
plt.colorbar(label='Solar Masses')
plt.title(r'FIRE simulation (100 km/s)')
plt.axis('off')
plt.savefig('/Users/mikezagar/Desktop/ENPH 455/Fire simulation @ 70 degs.jpeg')
plt.show()