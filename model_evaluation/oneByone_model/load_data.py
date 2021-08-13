import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import tracemalloc
import sys

def format_bytes(size):
    power = 2 ** 10
    n = 0
    power_labels = ['B', 'KB', 'MB', 'GB', 'TB']
    while size > power and n <= len(power_labels):
        size /= power
        n += 1
    return f"current used memory: {size} {power_labels[n]}"

def log_memory():
    snapshot = tracemalloc.take_snapshot()
    size = sum([stat.size for stat in snapshot.statistics('filename')])
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
    if 'rain' in path:
        #df = df + 50
        # Scale [0, 100]
        return min_max_scaler(0, 100, df.values)
        
    elif 'temp' in path:
        # Scale [10, 45]
        return min_max_scaler(10, 45, df.values)

    elif 'abs_wind' in path:
        df = np.where(df > 15, 15, df)
        return min_max_scaler(0, 15, df)
        
    elif 'wind' in path:
        # Scale [-10, 10]
        return min_max_scaler(-10, 10, df.values)

    elif 'humidity' in path:
        return min_max_scaler(0, 100, df.values)

def get_param_path(param_name: str, year, month, date):
    if 'rain' in param_name:
        return f'../../../data/rain_image/{year}/{month}/{date}'
    elif 'wind' in param_name:
        return f'../../../data/wind_image/{year}/{month}/{date}'
    elif 'temperature' in param_name:
        return f'../../../data/temp_image/{year}/{month}/{date}'
    elif 'humidity' in param_name:
        return f'../../../data/humidity_image/{year}/{month}/{date}'
    else:
        print(param_name, "is wrong or spell missing.")
        return

def chack_data_scale(data):
    if data.max() > 1 or data.min() < 0:
        return False
    else: return True


def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta
        
def create_time_list(year=2020, month=1, date=1):
    dts = [dt for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))]
    return dts


def load_data(params=['rain', 'humidity', 'temperature', 'wind']):
    tracemalloc.start()

    time_list = create_time_list()

    input_arr = []
    label_arr = []

    year = 2019
    monthes = ['10', '11']
    dates_list = []
    time_set = []

    for month in monthes:
        log_memory()
        dates = os.listdir(f'../../../data/rain_image/{year}/{month}')
        dates_list.append(dates)

        for date in dates:
            print("... Loading", date)
            params_path = {}
            if len(params) > 0:
                for par in params:
                    params_path[par] = get_param_path(param_name=par, year=year, month=month, date=date)
            
            time_subset = []
            for step in range(0, len(time_list) - 6, 6):
                file_names = [f'{dt.hour}-{dt.minute}.csv' for dt in time_list[step:step+12]]
                time_subset.append(file_names[6:])
                
                subset_arrs = []
                for file_name in file_names:
                    params_data = {}
                    # Load data
                    for par in params:
                        if 'wind' in par:
                            params_data['u_wind'] = load_csv_data(params_path[par] + f'/{file_name}'.replace('.csv', 'U.csv'))
                            params_data['v_wind'] = load_csv_data(params_path[par] + f'/{file_name}'.replace('.csv', 'V.csv'))
                            
                        else:
                            params_data[par] = load_csv_data(params_path[par] + f'/{file_name}')

                    # Check if data is valid
                    for key in params_data.keys():
                        if not chack_data_scale(params_data[key]):
                            print(year, month, date, file_name, key, " Has invalid scale.(x > 1 or x < 0)")
                            sys.exit()
                    
                    subset_arr = np.empty([50, 50, len(params_data.keys())])
                    for i in range(50):
                        for j in range(50):
                            for k, key in enumerate(params_data.keys()):
                                subset_arr[i, j, k] = params_data[key][i, j]
                    
                    subset_arrs.append(subset_arr)
                
                input_arr.append(subset_arrs[:6])
                label_arr.append(subset_arrs[6:])
        time_set.append(time_subset)

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, len(params_data.keys())])
    label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, len(params_data.keys())])

    data_config = {
        'year': year,
        'monthes': monthes,
        'dates': dates_list,
        'time': time_set
    }

    return input_arr, label_arr, data_config


if __name__ == '__main__':
    input_arr, label_arr, data_config = load_data(params=['rain', 'humidity', 'temperature', 'wind'])
    print(input_arr.shape, label_arr.shape)
    print(data_config)