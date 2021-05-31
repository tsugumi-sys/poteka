import os
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image

from load_data import load_data
from load_image_data import load_rainbow_rain_data, load_dense_rain_data

def rescale_arr(min_value, max_value, arr):
    return (max_value - min_value) * arr + min_value



def save_csv(data_arr, path: str):
    grid_lon, grid_lat = np.round(np.linspace(120.90, 121.50, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
    df = pd.DataFrame(data_arr, index=np.flip(grid_lat), columns=grid_lon)
    df.to_csv(path)
    print(path)


def make_prediction(model_name='model1'):
    print('-' * 80)
    print('This is prediction with multi variable trained model')
    print('-'*80)
    model = load_model(f'../../model/{model_name}/model.h5')
    X, y, data_config = load_data()
    preds = model.predict(X)
    print(preds.shape)

    year = data_config['year'] # str
    monthes = data_config['monthes'] # list
    dates = data_config['dates'] # list
    time_lists = data_config['time'] #list

    count = 0
    data_count = 0
    for month in monthes:
        for date in dates[count]:
            for time_list in time_lists[count]:
                time_count = 0
                for time in time_list:
                    path = '../../data/prediction_image/'
                    for item in [model_name, year, month, date]:
                        path += f'{item}/'
                        if not os.path.exists(path):
                            os.mkdir(path)

                    rain_arr = np.empty([50, 50])
                    print(preds[data_count][time_count])
                    for i in range(50):
                        for j in range(50):
                            rain_arr[i, j] = preds[data_count][time_count][i, j, 0]

                    rain_arr = np.where(rain_arr > 1, 1, rain_arr)
                    rain_arr = np.where(rain_arr < 0, 0, rain_arr)
                    rain_arr = rescale_arr(0, 150, rain_arr)
                    print(rain_arr.max(), rain_arr.min())
                   
                    save_csv(rain_arr, path + time)

                    time_count += 1
                data_count += 1
                print(data_count)
        count += 1

# image trained model
def make_image_trained_prediction(model_name='model1', img_color='rainbow'):
    # Background Image
    bg = Image.open('bg_rainbow.png').convert('RGBA')
    img_clear = Image.new('RGBA', bg.size, (255, 255, 255, 0))

    print('-' * 80)
    print(f'This is making prediction of {img_color} colored images.')
    print('-'*80)
    model = load_model(f'../../model/train_with_image/{model_name}/model.h5')
    if img_color == 'rainbow':
        X, y, data_config = load_rainbow_rain_data()
    else:
        X, y, data_config = load_dense_rain_data()
    preds = model.predict(X)
    print(preds.shape)

    year = data_config['year'] # str
    monthes = data_config['monthes'] # list
    dates = data_config['dates'] # list
    time_lists = data_config['time'] #list

    count = 0
    data_count = 0
    for month in monthes:
        for date in dates[count]:
            for time_list in time_lists[count]:
                time_count = 0
                for time in time_list:
                    path = '../../data/prediction_image/train_with_image/60min_'
                    for item in [model_name, year, month, date]:
                        path += f'{item}/'
                        if not os.path.exists(path):
                            os.mkdir(path)
                    print(preds[data_count][time_count].max(), preds[data_count][time_count].min(), preds[data_count][time_count].shape)
                    img_arr = preds[data_count][time_count]
                    print(img_arr.max(), img_arr.min())

                    # If img_arr scale is [-1, 1]
                    # img_arr = np.where(img_arr > 1, 1, img_arr)
                    # img_arr = np.where(img_arr < -1, -1, img_arr)
                    # img_arr = img_arr * 128 + 128
                    
                    # If img_arr scale is [0, 1]
                    img_arr = np.where(img_arr > 1, 1, img_arr)
                    img_arr = np.where(img_arr < 0, 0, img_arr)
                    img_arr = img_arr * 255


                    img_arr = img_arr.astype(np.uint8)
                    img = Image.fromarray(img_arr)
                    img = img.resize((299, 493))
                    img.save(path + time)
                    img_clear.paste(img, (118, 77))
                    img_with_bg = Image.alpha_composite(img_clear, bg)
                    img_with_bg.save(path + time.replace('croped', ''))

                    time_count += 1
                data_count += 1
                print(data_count)
        count += 1

    
            




if __name__ == '__main__':
    make_image_trained_prediction(model_name='model5', img_color='rainbow')
    make_image_trained_prediction(model_name='model6', img_color='dense')
    #make_prediction(model_name="model1")