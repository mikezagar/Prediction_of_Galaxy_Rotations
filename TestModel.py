import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from scipy.ndimage import zoom
from tensorflow_addons.layers import GroupNormalization
from sklearn.metrics import r2_score

from Neural_Network import MCDropout

y_scaler = joblib.load('y_scaler.pkl')
data_2d = np.loadtxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/data.txt')
rings = np.loadtxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/rings.txt')
ring_vels = np.loadtxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/ring_vels.txt')

model1 = load_model('model.h5', custom_objects={'GroupNormalization': GroupNormalization, 'MCDropout': MCDropout})

scalar = MinMaxScaler()
normalized_data = scalar.fit_transform(data_2d)
train_data = normalized_data.reshape(9, 300, 300, 200)

def downsample_cube(cube, target_shape=(64, 61, 142)):
    factors = (
        target_shape[0] / cube.shape[0],
        target_shape[1] / cube.shape[1],
        target_shape[2] / cube.shape[2]
    )
    return zoom(cube, zoom=factors, order=1)  # order=1 = linear interpolation

train_data = train_data[6]
downsampled_data = np.stack(downsample_cube(train_data, (64, 61, 142)))
downsampled_data = downsampled_data[..., np.newaxis]

test_data = np.expand_dims(downsampled_data, axis=0)

print(downsampled_data.shape)
print(test_data.shape)

def mc_dropout_predict(model, x_input, num_samples=50):
    preds = [model(x_input, training=True).numpy() for _ in range(num_samples)]
    preds = np.array(preds)  # shape: (samples, batch, bins)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred.squeeze(), std_pred.squeeze()  # (50,), (50,)

y_pred = model1.predict(test_data)

y_pred = np.squeeze(y_pred, axis=0)

output = y_pred.reshape(-1, 50)
out = y_scaler.inverse_transform(output)
out = out.reshape(y_pred.shape)
smoothed_curve = savgol_filter(out, 9, 3)

loss_fn = tf.keras.losses.MeanSquaredError()
loss = loss_fn(y_true=ring_vels[6,:], y_pred=smoothed_curve).numpy()
rmse = np.sqrt(loss)

r2 = r2_score(ring_vels[6, :], smoothed_curve)

print("Mean squared error loss:", loss)
print("Root mean square error loss:", rmse)
print("R2 score:", r2)

plt.plot(rings[6,:], smoothed_curve, '-.', color='green', label='Predicted Curve')
plt.plot(rings[6,:], ring_vels[6,:], '--', color='blue', label='Actual Curve')
plt.title(f'Predicted Ring Velocity, ' + str(r2))
plt.ylabel('V [km/s]')
plt.xlabel('R["]')
plt.legend()
plt.savefig('/Users/mikezagar/Desktop/ENPH 455/Images/Validation set.png')
plt.show()


