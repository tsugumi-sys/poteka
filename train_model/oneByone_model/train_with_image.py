# import os
# import requests
# import numpy as np
# import pandas as pd
# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras import layers, callbacks, metrics, regularizers
# from sklearn.model_selection import train_test_split
# import tracemalloc
# import traceback

# from load_image_data import load_data, load_dense_rain_data
# from Models.DLWP.model import DLWP_ConvLSTM
# from Models.Keras_EG.model import Keras_EG
# from dotenv import load_dotenv
# from pathlib import Path

# dotenv_path = Path('../.env')
# load_dotenv(dotenv_path=dotenv_path)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# def send_line(msg):
#     line_notify_token = os.getenv('LINE_TOKEN')
#     line_notify_endpoint = 'https://notify-api.line.me/api/notify'
#     headers = {'Authorization': f'Bearer {line_notify_token}'}
#     data = {'message': f'message: {msg}'}
#     res = requests.post(line_notify_endpoint, headers, data)
#     return res.status_code

# def create_model(img_height=60, img_width=36, model_type='default'):
#     if model_type == 'default':
#         model = keras.Sequential([
#             layers.Input(
#                 shape=(None, img_height, img_width, 3)
#             ),
#             layers.ConvLSTM2D(
#                 filters=40, kernel_size=(3, 3), padding='same', return_sequences=True,
#             ),
#             layers.BatchNormalization(),
#             layers.ConvLSTM2D(
#                 filters=40, kernel_size=(3, 3), padding='same', return_sequences=True,
#             ),
#             layers.BatchNormalization(),
#             layers.ConvLSTM2D(
#                 filters=40, kernel_size=(3, 3), padding='same', return_sequences=True,
#             ),
#             layers.BatchNormalization(),
#             layers.Conv3D(
#                 filters=3, kernel_size=(3, 3, 3), padding='same', activation='relu'
#             )
#         ])

#         model.compile(
#             optimizer='adam',
#             loss='mse',
#             metrics=['mse']
#         )
#         model.summary()
    
#     elif model_type == 'DLWP':
#         model = DLWP_ConvLSTM()

#     elif model_type == 'KerasEG':
#         model = Keras_EG()
#     return model


# # Train with Image: rainbow colored
# def train_rainbow_image_model(model_name='model5'):
#     print('-' * 80)
#     print('This is training with dense colored images.')
#     print('-'*80)
#     X, y, data_config = load_data()
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)

#     early_stopping = callbacks.EarlyStopping(
#         #monitor='val_loss',
#         min_delta=0.001,
#         patience= 15,
#         restore_best_weights=True
#     )

#     model = create_model(model_type='KerasEG')
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_valid, y_valid),
#         epochs=500,
#         batch_size=32,
#         callbacks=[early_stopping],
#         verbose=1
#     )

#     save_path = f'../../model/train_with_image/{model_name}/'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)

#     hist = pd.DataFrame(history.history)
#     hist.to_csv(save_path + 'history.csv')
#     model.save(save_path + 'model.h5')
#     print('Model Successfully Saved')

# # Train with Image: dense coloered
# def train_dense_image_model(model_name='model6'):
#     print('-' * 80)
#     print('This is training with dense colored images.')
#     print('-'*80)
#     X, y, data_config = load_dense_rain_data()
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=11)

#     early_stopping = callbacks.EarlyStopping(
#         # monitor='val_loss',
#         min_delta=0.001,
#         patience= 15,
#         restore_best_weights=True
#     )
#     model = create_model(model_type='KerasEG')
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_valid, y_valid),
#         epochs=500,
#         batch_size=32,
#         callbacks=[early_stopping],
#         verbose=1
#     )

#     save_path = f'../../model/train_with_image/{model_name}/'
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)

#     hist = pd.DataFrame(history.history)
#     hist.to_csv(save_path + 'history.csv')
#     model.save(save_path + 'model.h5')
#     print('Model Successfully Saved')


# if __name__ == '__main__':
#     try:
#         train_rainbow_image_model(model_name='model5')
#         train_dense_image_model(model_name='model6')
#         send_line('Successfully Completed')
#     except:
#         send_line('Process has Stoped with some Error')
#         send_line(traceback.format_exc())
#         print(traceback.format_exc())
