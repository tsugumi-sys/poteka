import os
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
from load_data import load_data_RUV

# from load_image_data import load_rainbow_rain_data, load_dense_rain_data
# import sys
from pathlib import Path
from dotenv import load_dotenv

# PACKAGE_PARENT = "../../"
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# from train_model.Models.DLWP.util import custom_load_model

dotenv_path = Path("../../.env")
load_dotenv(dotenv_path=dotenv_path)


def send_line(msg):
    line_notify_token = os.getenv("LINE_TOKEN")
    line_notify_endpoint = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"message: {msg}"}
    res = requests.post(line_notify_endpoint, headers, data)
    return res.status_code


def rescale_arr(min_value, max_value, arr):
    return (max_value - min_value) * arr + min_value


def save_csv(data_arr, path: str):
    grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
    df = pd.DataFrame(data_arr, index=np.flip(grid_lat), columns=grid_lon)
    df.to_csv(path)
    print(path)


def make_prediction(model_name="model1", time_span=60):
    print("-" * 80)
    print("This is prediction with multi variable trained model")
    print("Model Name:", model_name)
    print("-" * 80)
    model = load_model(f"../../../model/seq2seq_model/{model_name}/model.h5")
    X, y, data_config = load_data_RUV()

    year = data_config["year"]  # str
    monthes = data_config["monthes"]  # list
    dates = data_config["dates"]  # list
    time_lists = data_config["time"]  # list

    preds = model.predict(X)
    print(preds.shape)

    count = 0
    data_count = 0
    for month in monthes:
        for date in dates[count]:
            for time_list in time_lists[count]:
                time_count = 0
                # data = X[data_count].reshape(1, 6, 50, 50, 3)

                for time in time_list:
                    path = f"../../../data/prediction_image/seq2seq_model/{time_span}min_"
                    for item in [model_name, year, month, date]:
                        path += f"{item}/"
                        if not os.path.exists(path):
                            os.mkdir(path)

                    rain_arr = np.empty([50, 50])
                    pred = preds[data_count][time_count]
                    print(pred.shape)
                    # preds = model.predict(data)[0][-1]
                    # data = np.append(data[0][1:], [preds], axis=0).reshape(1, 6, 50, 50, 3)

                    for i in range(50):
                        for j in range(50):
                            rain_arr[i, j] = pred[i, j, 0]

                    rain_arr = np.where(rain_arr > 1, 1, rain_arr)
                    rain_arr = np.where(rain_arr < 0, 0, rain_arr)
                    rain_arr = rescale_arr(0, 100, rain_arr)
                    print(rain_arr.max(), rain_arr.min())

                    save_csv(rain_arr, path + time)

                    time_count += 1
                data_count += 1
                print(data_count)
        count += 1


if __name__ == "__main__":
    make_prediction(model_name="ruv_model_baseline", time_span=60)
    make_prediction(model_name="ruv_model_optuned", time_span=60)
