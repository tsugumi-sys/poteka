import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import tracemalloc


def format_bytes(size):
    power = 2 ** 10
    n = 0
    power_labels = ["B", "KB", "MB", "GB", "TB"]
    while size > power and n <= len(power_labels):
        size /= power
        n += 1
    return f"current used memory: {size} {power_labels[n]}"


def log_memory():
    snapshot = tracemalloc.take_snapshot()
    size = sum([stat.size for stat in snapshot.statistics("filename")])
    print(format_bytes(size))


def min_max_scaler(min_value, max_value, arr):
    return (arr - min_value) / (max_value - min_value)


def rescale_arr(min_value, max_value, arr):
    return (max_value - min_value) * arr + min_value


def rescale_rain_arr_150To100(min_value, max_value, arr):
    arr = (max_value - min_value) * arr + min_value
    arr -= 50
    arr = np.where(arr > 100, 100, arr)
    arr = np.where(arr < 0, 0, arr)
    return arr


# return: ndarray
def load_csv_data(path: str):
    df = pd.read_csv(path, index_col=0)
    if "rain" in path:
        # df = df + 50
        # Scale [0, 150]
        return min_max_scaler(0, 100, df.values)

    elif "temp" in path:
        # Scale [10, 45]
        return min_max_scaler(10, 45, df.values)

    elif "abs_wind" in path:
        df = np.where(df > 15, 15, df)
        return min_max_scaler(0, 15, df)

    elif "wind" in path:
        # Scale [-10, 10]
        return min_max_scaler(-10, 10, df.values)


def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta


def create_time_list(year=2020, month=1, date=1):
    dts = [dt for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))]
    return dts


def load_data_RUV():
    tracemalloc.start()

    time_list = create_time_list()

    input_arr = []
    label_arr = []

    year = 2019
    monthes = ["10", "11"]
    dates_list = []
    time_set = []

    for month in monthes:
        log_memory()
        dates = os.listdir(f"../../../data/rain_image/{year}/{month}")
        dates_list.append(dates)

        for date in dates:
            print(date)
            rain_path = f"../../../data/rain_image/{year}/{month}/{date}"
            wind_path = f"../../../data/wind_image/{year}/{month}/{date}"
            time_subset = []
            if os.path.exists(rain_path) and os.path.exists(wind_path):
                rain_file_num = len(os.listdir(rain_path))
                # wind_file_num = len(os.listdir(wind_path))
                if rain_file_num == 288:
                    for step in range(0, len(time_list) - 6, 6):
                        next_step = step + 12
                        file_names = [f"{dt.hour}-{dt.minute}.csv" for dt in time_list[step:next_step]]
                        time_subset.append(file_names[6:])

                        subset_arrs = []
                        for file_name in file_names:
                            # Load data
                            rain_file_path = rain_path + f"/{file_name}"
                            u_wind_file_path = wind_path + f"/{file_name}".replace(".csv", "U.csv")
                            v_wind_file_path = wind_path + f"/{file_name}".replace(".csv", "V.csv")

                            # Create ndarray
                            rain_arr = load_csv_data(rain_file_path)
                            u_wind_arr = load_csv_data(u_wind_file_path)
                            v_wind_arr = load_csv_data(v_wind_file_path)
                            # print('Rain Data', rain_arr.max(), rain_arr.min())
                            # print('U Wind Data', u_wind_arr.max(), u_wind_arr.min())
                            # print('V Wind Data', v_wind_arr.max(), v_wind_arr.min())
                            subset_arr = np.empty([50, 50, 3])
                            for i in range(50):
                                for j in range(50):
                                    subset_arr[i, j, 0] = rain_arr[i, j]
                                    subset_arr[i, j, 1] = u_wind_arr[i, j]
                                    subset_arr[i, j, 2] = v_wind_arr[i, j]

                            subset_arrs.append(subset_arr)

                        input_arr.append(subset_arrs[:6])
                        label_arr.append(subset_arrs[6:])
        time_set.append(time_subset)

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
    label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 3])

    data_config = {"year": year, "monthes": monthes, "dates": dates_list, "time": time_set}

    return input_arr, label_arr, data_config


def load_data_RWT():
    tracemalloc.start()

    time_list = create_time_list()

    input_arr = []
    label_arr = []

    year = 2019
    monthes = ["10", "11"]
    dates_list = []
    time_set = []

    for month in monthes:
        log_memory()
        dates = os.listdir(f"../../../data/rain_image/{year}/{month}")
        dates_list.append(dates)

        for date in dates:
            print(date)
            rain_path = f"../../../data/rain_image/{year}/{month}/{date}"
            temp_path = f"../../../data/temp_image/{year}/{month}/{date}"
            abs_wind_path = f"../../../data/abs_wind_image/{year}/{month}/{date}"
            time_subset = []
            if os.path.exists(rain_path) and os.path.exists(temp_path) and os.path.exists(abs_wind_path):
                rain_file_num = len(os.listdir(rain_path))
                # temp_file_num = len(os.listdir(temp_path))
                # abs_wind_file_num = len(os.listdir(abs_wind_path))
                if rain_file_num == 288:
                    for step in range(0, len(time_list) - 6, 6):
                        next_step = step + 12
                        file_names = [f"{dt.hour}-{dt.minute}.csv" for dt in time_list[step:next_step]]
                        time_subset.append(file_names[6:])

                        subset_arrs = []
                        for file_name in file_names:
                            # Load data
                            # Load data
                            rain_file_path = rain_path + f"/{file_name}"
                            temp_file_path = temp_path + f"/{file_name}"
                            abs_wind_file_path = abs_wind_path + f"/{file_name}"

                            # Create ndarray
                            rain_arr = load_csv_data(rain_file_path)
                            temp_arr = load_csv_data(temp_file_path)
                            abs_wind_arr = load_csv_data(abs_wind_file_path)
                            subset_arr = np.empty([50, 50, 3])
                            for i in range(50):
                                for j in range(50):
                                    subset_arr[i, j, 0] = rain_arr[i, j]
                                    subset_arr[i, j, 1] = temp_arr[i, j]
                                    subset_arr[i, j, 2] = abs_wind_arr[i, j]

                            subset_arrs.append(subset_arr)

                        input_arr.append(subset_arrs[:6])
                        label_arr.append(subset_arrs[6:])
        time_set.append(time_subset)

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
    label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 3])

    data_config = {"year": year, "monthes": monthes, "dates": dates_list, "time": time_set}

    return input_arr, label_arr, data_config


def load_rain_data():
    tracemalloc.start()

    time_list = create_time_list()

    input_arr = []
    label_arr = []

    year = 2019
    monthes = ["11"]
    dates_list = []
    time_set = []
    for month in monthes:
        log_memory()
        dates = os.listdir(f"../../../data/rain_image/{year}/{month}")
        dates_list.append(dates)

        for date in dates:
            print(date)
            rain_path = f"../../../data/rain_image/{year}/{month}/{date}"
            time_subset = []
            if os.path.exists(rain_path):
                rain_file_num = len(os.listdir(rain_path))
                if rain_file_num == 288:
                    for step in range(0, len(time_list) - 6, 6):
                        next_step = step + 12
                        file_names = [f"{dt.hour}-{dt.minute}.csv" for dt in time_list[step:next_step]]
                        time_subset.append(file_names[6:])

                        subset_arrs = []
                        for file_name in file_names:
                            # Load data
                            rain_file_path = rain_path + f"/{file_name}"

                            # Create ndarray
                            rain_arr = load_csv_data(rain_file_path)
                            subset_arr = np.empty([50, 50, 3])
                            for i in range(50):
                                for j in range(50):
                                    subset_arr[i, j, 0] = rain_arr[i, j]
                                    subset_arr[i, j, 1] = 0
                                    subset_arr[i, j, 2] = 0

                            subset_arrs.append(subset_arr)

                        input_arr.append(subset_arrs[:6])
                        label_arr.append(subset_arrs[6:])
            time_set.append(time_subset)

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
    label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 3])

    data_config = {"year": year, "monthes": monthes, "dates": dates_list, "time": time_set}

    return input_arr, label_arr, data_config


if __name__ == "__main__":
    input_arr, label_arr, data_config = load_data_RUV()
    print(input_arr.shape, label_arr.shape)
    print(data_config)
