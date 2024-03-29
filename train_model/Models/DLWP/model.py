from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Input, ConvLSTM2D, ZeroPadding3D, MaxPooling3D, UpSampling3D, Conv3D, BatchNormalization
from .custom import PeriodicPadding3D


def DLWP_ConvLSTM(img_height=60, img_width=36):
    model = Sequential(
        [
            Input(shape=(None, img_height, img_width, 3)),
            PeriodicPadding3D(padding=(0, 0, 2), data_format="channels_first"),
            ZeroPadding3D(padding=(0, 2, 0), data_format="channels_first"),
            ConvLSTM2D(
                filters=32,
                kernel_size=3,
                dilation_rate=2,
                padding="same",
                data_format="channels_first",
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1.0e-4),
            ),
            BatchNormalization(),
            MaxPooling3D(pool_size=(1, 1, 1), data_format="channels_first"),
            PeriodicPadding3D(padding=(0, 0, 1), data_format="channels_first"),
            ZeroPadding3D(padding=(0, 1, 0), data_format="channels_first"),
            ConvLSTM2D(
                filters=32,
                kernel_size=3,
                dilation_rate=1,
                padding="same",
                data_format="channels_first",
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1.0e-4),
            ),
            BatchNormalization(),
            MaxPooling3D(pool_size=(1, 1, 1), data_format="channels_first"),
            PeriodicPadding3D(padding=(0, 0, 1), data_format="channels_first"),
            ZeroPadding3D(padding=(0, 1, 0), data_format="channels_first"),
            ConvLSTM2D(
                filters=64,
                kernel_size=3,
                dilation_rate=1,
                padding="same",
                data_format="channels_first",
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1.0e-4),
            ),
            BatchNormalization(),
            UpSampling3D(size=(1, 1, 1), data_format="channels_first"),
            PeriodicPadding3D(padding=(0, 0, 1), data_format="channels_first"),
            ZeroPadding3D(padding=(0, 1, 0), data_format="channels_first"),
            ConvLSTM2D(
                filters=32,
                kernel_size=3,
                dilation_rate=1,
                padding="same",
                data_format="channels_first",
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1.0e-4),
            ),
            BatchNormalization(),
            UpSampling3D(size=(1, 1, 1), data_format="channels_first"),
            PeriodicPadding3D(padding=(0, 0, 2), data_format="channels_first"),
            ZeroPadding3D(padding=(0, 2, 0), data_format="channels_first"),
            ConvLSTM2D(
                filters=64,
                kernel_size=3,
                dilation_rate=2,
                padding="same",
                data_format="channels_first",
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=regularizers.l2(1.0e-4),
            ),
            BatchNormalization(),
            PeriodicPadding3D(padding=(0, 0, 2), data_format="channels_first"),
            ZeroPadding3D(padding=(0, 2, 0), data_format="channels_first"),
            ConvLSTM2D(
                filters=64, kernel_size=5, dilation_rate=1, padding="same", data_format="channels_last", activation="tanh", return_sequences=True
            ),
            BatchNormalization(),
            Conv3D(filters=3, kernel_size=(3, 3, 3), padding="same", activation="relu"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


if __name__ == "__main__":
    DLWP_ConvLSTM()
