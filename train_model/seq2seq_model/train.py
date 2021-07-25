import os
import requests
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, callbacks, metrics, regularizers
from sklearn.model_selection import train_test_split
import tracemalloc
import traceback
import optuna
import mlflow
from mlflow import pyfunc
import mlflow.tensorflow

from load_data import load_data_RUV, load_rain_data, load_data_RWT
import sys
sys.path.insert(0, '..')
from Models.DLWP.model import DLWP_ConvLSTM
from Models.Keras_EG.model import Keras_EG, Modified_Keras_EG, Modified_Keras_EG_OnebyOne

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../../.env')
load_dotenv(dotenv_path=dotenv_path)

def send_line(msg):
    line_notify_token = os.getenv('LINE_TOKEN')
    line_notify_endpoint = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {msg}'}
    res = requests.post(line_notify_endpoint, headers, data)
    return res.status_code

def create_model(params):
    # Input Shape Pramameters
    HEIGHT = 50
    WIDTH = 50

    # Parameters
    filters = params['filters']
    adam_learning_rate = params['adam_learning_rate']

    
    inp = layers.Input(shape=(None, HEIGHT, WIDTH, 3))
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(5, 5),
        padding='same',
        return_sequences=True,
        activation='relu',
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='relu',
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=filters,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='relu',
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=adam_learning_rate), metrics=['mse', metrics.RootMeanSquaredError()]
    )
    model.summary()
    return model

# Multi Variable Model
def main():
    model_name = 'ruv_model_optuned'
    print('-'*60)
    print(f'Model Name: {model_name}')
    print('-'*60)
    params = {
        'filters': 44,
        'adam_learning_rate': 0.0008217,
    }
    keras.backend.clear_session()

    mlflow.set_experiment('Seq2Seq_ConvLSTM')
    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run(run_name=model_name):
        X, y = load_data_RUV()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)
        model = create_model(params)
        mlflow.log_params(params)

        early_stopping = callbacks.EarlyStopping(
            min_delta= 0.001,
            patience= 20,
            restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=500,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=1
        )

        score = model.evaluate(X_valid, y_valid, verbose=0)

    save_path = f'../../../model/seq2seq_model/{model_name}/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    hist = pd.DataFrame(history.history)
    hist.to_csv(save_path + 'history.csv')
    model.save(save_path + 'model.h5')
    print('Model Successfully Saved')


if __name__ == '__main__':
    try:
        main()
        send_line('Successfully Completed')
    except:
        send_line('Process has Stoped with some Error')
        send_line(traceback.format_exc())
        print(traceback.format_exc())
