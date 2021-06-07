import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def Keras_EG(img_height=60, img_width=36):
    inp = layers.Input(shape=(None, img_height, img_width, 3))
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(5, 5),
        padding='same',
        return_sequences=True,
        activation='relu',
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='relu'
    )(x)
    x = layers.Conv3D(
        filters=3, kernel_size=(3, 3, 3), activation='sigmoid', padding='same'
    )(x)

    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
    )
    model.summary()
    return model

if __name__ == '__main__':
    Keras_EG()()