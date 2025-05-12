import os
import glob
from astropy.io import fits

from Find_Curve import *
from scipy.signal import savgol_filter

folder_path = "/Users/mikezagar/Desktop/ENPH 455/Training Image Data"

# Create a list of all .fits files in the folder
fits_files = sorted(glob.glob(os.path.join(folder_path, "*.fits")))

# Prepare a list to store data arrays
data_list = []
rings = []
ring_vels = []

for filename in fits_files:
    radius, rotation_curve = fit_tilted_rings(filename)
    rings.append(radius)
    smoothed_curve = savgol_filter(rotation_curve, 7, 3)
    ring_vels.append(smoothed_curve)

    with fits.open(filename) as hdul:
        # Assuming the image data is in the primary HDU (hdul[0])
        image_data = hdul[0].data

        # Append the FITS data array to the list
        data_list.append(image_data)

# Convert the list of arrays to a single NumPy array
# If each image is the same shape, this produces a 3D array
# with shape (num_images, image_height, image_width).
all_data = np.array(data_list)

data_2d = all_data.reshape(all_data.shape[0], -1)

print("Number of images loaded:", len(data_2d))
print("Shape of the stacked array:", data_2d.shape)

np.savetxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/data.txt', data_2d)
np.savetxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/rings.txt', rings)
np.savetxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/ring_vels.txt', ring_vels)
