import tensorflow
from tensorflow.compat.v1 import keras as tfk
from tensorflow import cast
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import ZeroPadding2D, ZeroPadding3D, LocallyConnected2D, Lambda, Layer, InputSpec, concatenate
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.utils import conv_utils
#import tensorflow.keras.utils as conv_utils
#from tensorflow.python.keras.engine.base_layer import InputSec
from tensorflow.keras import activations, initializers, regularizers, constraints#
import numpy as np


# ================================= #
# Keras padding layers
# ================================= #
class PeriodicPadding2D(ZeroPadding2D):
    """ Periodic-padding layer for 2D input (e.g. image).

    This layer can add periodic rows and columns at the top, bottom, left, right side of image tensor.
    Adapted from keras.layers.ZeroPadding2D by @jweyn
    # Arguments
      padding: int, or tuple of 2 ints, or tuple of tuples of 2 ints.
        - If int: the same symmetric padding 
            is applied to height and width.
        - If tuple of 2 ints:
            interpreted as two different
            symmetric padidng values for height and width:
            `(symmetric_height_pad, symmetric_width_pad)`.
        - If tuple of tuples of 2 ints:
            interpreted as
            `((top_pad, bottom_pad, (left_pad, right_pad)))`
      data_format: A string,
        one of "channels_last" or "channels_first".
        The ordering of the dimentions in the inputs.
        `"channels_last"` correspands to inputs with shape
        `(batch, height, width, channels)` while `"channels_first"`
        corresponds to input with shape
        `(batch, channelsm height, width)`.
        If defaults to the `image_data_format` value found in your
        Keras config file as `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last"

    # Input shape
        4D tensor with shape:
            - If `data_format` is "channels_last":
                `(batch, rows, cols, channels)`
            - If `data_format` is "channels_first":
                `(batch, channels, rows, cols)`

    # Output shape
        4D tensor with shape:
            - If `data_format` is "channels_last":
                `(batch, padded_rows, padded_cols, channels)`
            - If `data_format` is "channels_first"
                `(batch, channels, padded_rows, padded_cols)`
    """

    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super(PeriodicPadding2D, self).__init__(padding=padding, data_format=data_format, **kwargs)

    def call(self, inputs):
        shape = inputs.shape
        if self.data_format == 'channels_first':
            top_slice = slice(shape[2] - self.padding[0][0], shape[2])
            bottom_slice = slice(0, self.padding[0][1])
            left_slice = slice(shape[3] - self.padding[1][0], shape[3])
            right_slice = slice(0, self.padding[1][1])
            # Pad the horizontal
            outputs = concatenate([inputs[:, :, :, left_slice], inputs, inputs[:, :, :, right_slice]], axis=3)
            # Pad the vertical
            outputs = concatenate([outputs[:, :, top_slice], outputs, outputs[:, :, bottom_slice]], axis=2)
        else:
            top_slice = slice(shape[1] - self.padding[0][0], shape[1])
            bottom_slice = slice(0, self.padding[0][1])
            left_slice = slice(shape[2] - self.padding[1][0], shape[2])
            right_slice = slice(0, self.padding[1][1])
            # Pad the horizontal
            outputs = concatenate([input[:, :, left_slice], inputs, inputs[:, :, right_slice]], axis=2)
            # Pad the vertical
            outputs = concatenate([outputs[:, top_slice], outputs, outputs[:, bottom_slice]], axis=1)
        return outputs


class PeriodicPadding3D(ZeroPadding3D):
    """ Periodic-padding layer for 3D input (e.g. image).
    This layer can add periodic rows, columns, and depth to an image tensor.
    Adapted from keras.layers.ZeroPadding3D by @jweyn

    # Arguments
        padding: int, or tuple f 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width
            - If tuple of 3 ints:
                interpreted as two different symmetric padding values for height and width
                `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
            - If tuple of 3 tuples of 2 ints:
                interpreted as
                `((left_dim1_pad, right_dim1_pad),
                  (left_dim2_pad, right_dim2_pad),
                  (left_dim3_pad, right_dim3_pad))`
        data_format: A string
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            While `"channels_first"` corresponds to inputs shape with
            `((batch, channels, spatial_dim1, spatial_dim2, spatial_dim3))`.
            If defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`
            If yo never set it, then it will be "channels_last".

    # Input shape
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`
        - If `data_format` is `"channels_first"`:
            `(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`
    
    """

    def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
        super(PeriodicPadding3D, self).__init__(padding=padding, data_format=data_format, **kwargs)

    def call(self, inputs):
        shape = inputs.shape
        if self.data_format == 'channels_first':
            low_slice = slice(shape[2] - self.padding[0][0], shape[2])
            high_slice = slice(0, self.padding[0][1])
            top_slice = slice(shape[3] - self.padding[1][0], shape[3])
            bottom_slice = slice(0, self.padding[1][1])
            left_slice = slice(shape[4] - self.padding[2][0], shape[4])
            right_slice = slice(0, self.padding[2][1])
            # Pad the horizontal
            outputs = concatenate([inputs[:, :, :, :, left_slice], inputs, inputs[:, :, :, :, right_slice]], axis=4)
            # Pad the vertical
            outputs = concatenate([outputs[:, :, :, top_slice], outputs, outputs[:, :, :, bottom_slice]], axis=3)
            # Pad the depth
            outputs = concatenate([outputs[:, :, low_slice], outputs, outputs[:, :, high_slice]], axis=2)
        else:
            low_slice = slice(shape[1] - self.padding[0][0], shape[1])
            high_slice = slice(0, self.padding[0][1])
            top_slice = slice(shape[2] - self.padding[1][0], shape[2])
            bottom_slice = slice(0, self.padding[1][1])
            left_slice = slice(shape[3] - self.padding[2][0], shape[3])
            right_slice = slice(0, self.padding[2][1])
            # Pad the horizontal
            outputs = concatenate([inputs[:, :, :, left_slice], inputs, inputs[:, :, :, right_slice]], axis=3)
            # Pad the vertical
            outputs = concatenate([outputs[:, :, top_slice], outputs, outputs[:, :, bottom_slice]], axis=2)
            # Pad the depth
            outputs = concatenate([outputs[:, low_slice], outputs, outputs[:, high_slice]], axis=1)
        return outputs