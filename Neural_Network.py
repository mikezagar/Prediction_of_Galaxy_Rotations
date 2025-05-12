import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from keras.utils.vis_utils import plot_model
from tensorflow.keras.losses import Huber, LogCosh
from tensorflow.keras.regularizers import l2
from tensorflow_addons.layers import GroupNormalization

class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)  # Always use dropout

def build_3D_cnn(input_shape = (64, 61, 142, 1), num_bins = 50):
    model = Sequential()

    model.add(layers.Conv3D(
        filters = 32,
        kernel_size = (3, 3, 3),
        padding = 'same',
        strides = (1, 1, 1),
        input_shape = input_shape
    ))
    model.add(GroupNormalization(groups = 8, axis = -1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(layers.Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        padding='same',
        strides=(2, 2, 2)
    ))
    model.add(GroupNormalization(groups = 8, axis = -1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(layers.Conv3D(
        filters=128,
        kernel_size=(3, 3, 3),
        padding='same',
        strides=(2, 2, 2)
    ))
    model.add(GroupNormalization(groups = 8, axis = -1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(layers.Conv3D(
        filters=256,
        kernel_size=(3, 3, 3),
        padding='same',
        strides=(2, 2, 2)
    ))
    model.add(GroupNormalization(groups=8, axis=-1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(layers.GlobalAveragePooling3D())
    model.add(layers.Reshape((1,256)))
    model.add(LSTM(64, return_sequences=False))
    model.add(GroupNormalization(groups = 8, axis = -1))

    model.add(layers.Dense(512, kernel_regularizer=l2(0.001)))
    model.add(GroupNormalization(groups = 8, axis = -1))
    model.add(layers.Activation('relu'))
    model.add(Dropout(0.3))

    model.add(layers.Dense(256, kernel_regularizer = l2(0.001)))
    model.add(GroupNormalization(groups = 8, axis = -1))
    model.add(layers.Activation('relu'))
    model.add(Dropout(0.3))

    model.add(layers.Dense(128, kernel_regularizer=l2(0.001)))
    model.add(GroupNormalization(groups=8, axis=-1))
    model.add(layers.Activation('relu'))
    model.add(Dropout(0.3))

    model.add(layers.Dense(num_bins, activation = 'tanh'))

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = 0.001, decay_steps = 10000, alpha=0.1
    )
    model.compile(
        optimizer = AdamW(learning_rate = lr_schedule, weight_decay = 1e-4),
        loss = LogCosh(),
        metrics = ['mae']
    )

    return model

model = build_3D_cnn()

plot_model(model, to_file = 'model.png')

layer_info = []
for layer in model.layers:
    layer_info.append({
        'Layer Type': layer.__class__.__name__,
        'Output Shape': layer.output_shape,
        'Parameter #': layer.count_params()
    })

df = pd.DataFrame(layer_info)

df.to_csv('/Users/mikezagar/Desktop/ENPH 455/layer_info.csv')