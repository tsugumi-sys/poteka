import os
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, callbacks, metrics
from sklearn.model_selection import train_test_split
import traceback
import mlflow
import mlflow.tensorflow
from common.send_info import send_line
from common.ObO_data_loader import load_selected_data

# from load_data import load_selected_data

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def create_model(params, feature_num):
    # Input Shape Pramameters
    HEIGHT = 50
    WIDTH = 50

    # Parameters
    filters = params["filters"]
    adam_learning_rate = params["adam_learning_rate"]
    activation = params["activation"]

    inp = layers.Input(shape=(None, HEIGHT, WIDTH, feature_num))
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation=activation,
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation=activation,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation=activation,
    )(x)
    x = layers.BatchNormalization()(x)
    x = (layers.ConvLSTM2D(filters=feature_num, kernel_size=3, padding="same", return_sequences=False, activation="sigmoid")(x),)

    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=adam_learning_rate),
        metrics=["mse", metrics.RootMeanSquaredError()],
    )
    model.summary()
    return model


# Multi Variable Model
def main():

    # ruvthpp_optuna & baseline
    # filters = 38
    # lr = 0.00039

    # ruvth_optuna & baseline
    # filters = 38
    # lr = 0.0005

    model_name = "ruvthpp_optuna_no_earystop"
    params = {"filters": 38, "adam_learning_rate": 0.0005, "activation": "relu"}
    input_params = ["rain", "humidity", "temperature", "u_wind", "v_wind", "seaLevel_pressure", "station_pressure"]

    keras.backend.clear_session()

    mlflow.set_experiment("ConvLSTM")
    mlflow.tensorflow.autolog(every_n_iter=1)
    print("-" * 60)
    print(model_name)
    print("-" * 60)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_params(dict((f"parameter{i}", input_params[i]) for i in range(len(input_params))))

        X, y = load_selected_data(params=input_params)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)
        model = create_model(params, feature_num=len(input_params))

        early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=20, restore_best_weights=True)
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=1000,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=1,
        )

        score = model.evaluate(X_valid, y_valid, verbose=0)

    save_path = f"../../../model/oneByone_model/{model_name}/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    hist = pd.DataFrame(history.history)
    hist.to_csv(save_path + "history.csv")
    model.save(save_path + "model.h5")
    print("Model Successfully Saved")
    print(score)
    return score[-1]


if __name__ == "__main__":
    try:
        main()
        send_line("Successfully Completed")
    except:
        send_line("Process has Stoped with some Error")
        send_line(traceback.format_exc())
        print(traceback.format_exc())
