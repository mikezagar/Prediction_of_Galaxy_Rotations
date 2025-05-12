import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import GroupNormalization
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import joblib
from astropy import units as u
from spectral_cube import SpectralCube
from sklearn.metrics import r2_score

file_path = '/Users/mikezagar/Desktop/ENPH 455/WALLABY/WALLABY_J103915-301757_Hydra_TR2_cube.fits'
mask = fits.open('/Users/mikezagar/Desktop/ENPH 455/WALLABY/WALLABY_J103915-301757_Hydra_TR2_mask.fits')
mask_data = mask[0].data

hdu = SpectralCube.read(file_path)
masked_cube = hdu.with_mask(mask_data)

hdul = masked_cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value= 1.42 * u.GHz)
X = hdul.unmasked_data[:].value

from mpl_toolkits.axes_grid1 import make_axes_locatable
vel_curve_file = '/Users/mikezagar/Desktop/ENPH 455/J103915-301757/WALLABY_J103915-301757_2drings.txt'
radius, vsys, vrot, pa, incl  = np.loadtxt(vel_curve_file, usecols = (1, 2, 3, 5, 6), unpack=True)

y_scaler = joblib.load('y_scaler.pkl')
model1 = load_model('model.h5', custom_objects={'GroupNormalization': GroupNormalization})
model2 = load_model('model1.h5', custom_objects={'GroupNormalization': GroupNormalization})

X = np.transpose(X, (2, 1, 0))
print(X.shape)

data_2d = X.reshape(X.shape[0], -1)

scalar = MinMaxScaler()
X_scaled = scalar.fit_transform(data_2d)
X_scaled = X_scaled.reshape(64, 61, 142)
test_data = np.expand_dims(X_scaled, axis=0)

y_pred = (model1.predict(test_data) + model2.predict(test_data))/2
#y_pred = model1.predict(test_data)
y_pred = np.squeeze(y_pred, axis=0)

output = y_pred.reshape(-1, 50)
out = y_scaler.inverse_transform(output)
out = out.reshape(y_pred.shape)
smoothed_curve = savgol_filter(out, 9, 3) + 100

r2 = r2_score(vrot[:11], smoothed_curve[:11])
print("R2 Score: ", r2)

plt.plot(radius[:11], smoothed_curve[:11], '-.', color='green', label="True Curve")
plt.plot(radius[:11], vrot[:11], '--', color='blue', label="Predicted Curve")
plt.title("Predicted velocity curve from WALLABY dataset")
plt.xlabel('R["]')
plt.ylabel('V [km/s]')
plt.legend()
plt.show()



