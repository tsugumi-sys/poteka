import os
import requests
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, callbacks, metrics
from sklearn.model_selection import train_test_split
import traceback
import optuna
import mlflow
import mlflow.tensorflow
from load_data import load_data_RUV
from dotenv import load_dotenv
from pathlib import Path

# import sys
# sys.path.insert(0, '..')
# from Models.DLWP.model import DLWP_ConvLSTM
# from Models.Keras_EG.model import Keras_EG, Modified_Keras_EG, Modified_Keras_EG_OnebyOne

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

dotenv_path = Path("../../.env")
load_dotenv(dotenv_path=dotenv_path)


def send_line(msg):
    line_notify_token = os.getenv("LINE_TOKEN")
    line_notify_endpoint = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"message: {msg}"}
    res = requests.post(line_notify_endpoint, headers, data)
    return res.status_code


def create_model(trial):
    # Input Shape Pramameters
    HEIGHT = 50
    WIDTH = 50

    # Parameters
    filters = trial.suggest_int("filters", 16, 64)
    adam_learning_rate = trial.suggest_loguniform("adam_learning_rate", 1e-5, 1e-1)

    inp = layers.Input(shape=(None, HEIGHT, WIDTH, 3))
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=1, kernel_size=(3, 3, 3), padding="same", activation="sigmoid")(x)

    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=adam_learning_rate),
        metrics=[
            "mse",
            metrics.RootMeanSquaredError(),
        ],
    )
    model.summary()
    return model


# Multi Variable Model
def objective(trial):
    keras.backend.clear_session()

    mlflow.set_experiment("Optuna_Selected_ConvLSTM")
    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run():
        model = create_model(trial)

        mlflow.log_params(trial.params)

        early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=500, batch_size=16, callbacks=[early_stopping], verbose=1)

        score = model.evaluate(X_valid, y_valid, verbose=0)
    return score[1]


if __name__ == "__main__":
    try:
        X, y = load_data_RUV(dataType="selected")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, gc_after_trial=True)

        print("Number of finished trials", len(study.trials))
        print("Best Trials: ")
        trial = study.best_trial

        print(" Value: ", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("  {}: {}".format(key, value))

        send_line("Successfully Completed")
    except:
        send_line("Process has Stoped with some Error")
        send_line(traceback.format_exc())
        print(traceback.format_exc())
