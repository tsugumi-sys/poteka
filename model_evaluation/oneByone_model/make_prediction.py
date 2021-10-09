import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# from load_data import load_data
from common.ObO_data_loader import load_valid_data
from common.send_info import send_line
from common.utils import rescale_arr


def save_csv(data_arr, path: str):
    grid_lon, grid_lat = np.round(np.linspace(120.90, 121.150, 50), 3), np.round(np.linspace(14.350, 14.760, 50), 3)
    df = pd.DataFrame(data_arr, index=np.flip(grid_lat), columns=grid_lon)
    df.to_csv(path)
    print(path)


def make_prediction(
    model_name="model1",
    time_span=60,
    params=["rain", "humidity", "temperature", "abs_wind", "seaLevel_pressure", "station_pressure", "u_wind", "v_wind"],
):

    model = load_model(f"../../../model/oneByone_model/{model_name}/model.h5")
    print("-" * 80)
    print("This is prediction with multi variable trained model")
    print(f"Model Name: {model_name}")
    print("-" * 80)
    X, y, data_config = load_valid_data(params=params)
    feature_num = len(params)

    year = data_config["year"]  # str
    monthes = data_config["monthes"]  # list
    dates = data_config["dates"]  # list
    time_lists = data_config["time"]  # list

    count = 0
    data_count = 0
    for month in monthes:
        for date in dates[count]:
            for time_list in time_lists[count]:
                time_count = 0
                data = X[data_count].reshape(1, 6, 50, 50, feature_num)

                for time in time_list:
                    path = f"../../../data/prediction_image/oneByone_model/{time_span}min_"
                    for item in [model_name, year, month, date]:
                        path += f"{item}/"
                        if not os.path.exists(path):
                            os.mkdir(path)

                    rain_arr = np.empty([50, 50])
                    preds = model.predict(data)[0]
                    data = np.append(data[0][1:], [preds], axis=0).reshape(1, 6, 50, 50, feature_num)

                    for i in range(50):
                        for j in range(50):
                            rain_arr[i, j] = preds[i, j, 0]

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
    make_prediction(model_name="ruvth_baseline", time_span=60, params=["rain", "u_wind", "v_wind", "temperature", "humidity"])
    make_prediction(model_name="ruvth_optuna", time_span=60, params=["rain", "u_wind", "v_wind", "temperature", "humidity"])
    # make_prediction(model_name='ruv_model_selectedData_optuned', time_span=60, params=['rain', 'u_wind', 'v_wind'])
    make_prediction(
        model_name="ruvthpp_baseline",
        time_span=60,
        params=["rain", "humidity", "temperature", "u_wind", "v_wind", "seaLevel_pressure", "station_pressure"],
    )
    make_prediction(
        model_name="ruvthpp_optuna",
        time_span=60,
        params=["rain", "humidity", "temperature", "u_wind", "v_wind", "seaLevel_pressure", "station_pressure"],
    )
    # make_prediction(model_name='rwthpp_baseline', time_span=60, params=['rain', 'humidity', 'temperature', 'abs_wind', 'seaLevel_pressure','station_pressure'])
    # make_prediction(model_name='rwthpp_optuna', time_span=60, params=['rain', 'humidity', 'temperature', 'abs_wind', 'seaLevel_pressure','station_pressure'])
    send_line("Succesfully Ended")
