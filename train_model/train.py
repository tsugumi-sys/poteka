import os
import requests
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
import tracemalloc
import traceback

from load_data import load_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def send_line(msg):
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_endpoint = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {msg}'}
    res = requests.post(line_notify_endpoint, headers, data)
    return res.status_code

def create_model():
    model = keras.Sequential([
        layers.Input(
            shape=(None, 50, 50, 3)
        ),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding='same', return_sequences=True
        ),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding='same', return_sequences=True
        ),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding='same', return_sequences=True
        ),
        layers.Conv3D(
            filters=3, kernel_size=(3, 3, 3), padding='same'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )
    model.summary()
    return model

def train_model(model_name='model1'):
    X, y = load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)

    early_stopping = callbacks.EarlyStopping(
        min_delta= 0.0001,
        patience= 20,
        restore_best_weights=True
    )
    model = create_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    save_path = f'../../model/{model_name}/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    hist = pd.DataFrame(history.history)
    hist.to_csv(save_path + 'history.csv')
    model.save(save_path + 'model.h5')
    print('Model Successfully Saved')

if __name__ == '__main__':
    try:
        train_model()
        send_line('Successfully Completed')
    except:
        send_line('Process has Stoped with some Error')
        send_line(traceback.format_exc())
        print(traceback.format_exc())
