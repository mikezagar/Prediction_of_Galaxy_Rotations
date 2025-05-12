import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from scipy.ndimage import zoom

from Neural_Network import build_3D_cnn

data_2d = np.loadtxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/data.txt')
rings = np.loadtxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/rings.txt')
ring_vels = np.loadtxt('/Users/mikezagar/Desktop/ENPH 455/Training Data/ring_vels.txt')

model = build_3D_cnn()
model.summary()
'''
def add_gaussian_noise(data, mean=0.0, stddev=0.05):
    noise = np.random.normal(mean, stddev, data.shape)
    return data + noise'''

def augment_3D_data(datacube):
    # Random flipping
    if np.random.rand() > 0.5:
        datacube = np.flip(datacube, axis=0)  # Flip along the depth axis
    if np.random.rand() > 0.5:
        datacube = np.flip(datacube, axis=1)  # Flip along height

    # Random rotation
    if np.random.rand() > 0.5:
        datacube = np.rot90(datacube, k=np.random.choice([1, 2, 3]), axes=(0, 1))  # Rotate on RA-Dec plane
    return datacube

def downsample_cube(cube, target_shape=(64, 61, 142)):
    factors = (
        target_shape[0] / cube.shape[0],
        target_shape[1] / cube.shape[1],
        target_shape[2] / cube.shape[2]
    )
    return zoom(cube, zoom=factors, order=1)  # order=1 = linear interpolation

scalar = MinMaxScaler(feature_range=(0, 1))
normalized_data = scalar.fit_transform(data_2d)

y_scaler = MinMaxScaler(feature_range=(0, 1))
ring_vels = y_scaler.fit_transform(ring_vels)

train_data = normalized_data.reshape(9, 300, 300, 200)
train_data = np.delete(train_data, [6], axis=0)
print("Training data shape: ", train_data.shape)

ring_vels = np.delete(ring_vels, [6], axis=0)
print("Target data shape:", ring_vels.shape)

train_data_augmented = []
y_train = []
augmentation_factor = 2

for i in range(len(train_data)):
    train_data_augmented.append(train_data[i])
    y_train.append(ring_vels[i])

    for _ in range(augmentation_factor):
        new_sample = augment_3D_data(train_data[i])
        train_data_augmented.append(new_sample)
        y_train.append(ring_vels[i])

train_data_augmented = np.array(train_data_augmented, dtype=np.float32).reshape(24, 300, 300, 200)
ring_vel = np.array(y_train, dtype=np.float32).reshape(24, 50)

print("Augmented Training data shape:", train_data_augmented.shape)

downsampled_data = np.stack([downsample_cube(c, (64, 61, 142)) for c in train_data_augmented])
downsampled_data = downsampled_data[..., np.newaxis]

train_dataset = tf.data.Dataset.from_tensor_slices((downsampled_data, ring_vel))
train_dataset = (
    train_dataset
    .batch(64)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
early_stopping = EarlyStopping(monitor='mae', patience=10, restore_best_weights=True)
model.fit(train_dataset, epochs=200, callbacks=[early_stopping])

model.save('model1.h5')
joblib.dump(y_scaler, 'y_scaler.pkl')