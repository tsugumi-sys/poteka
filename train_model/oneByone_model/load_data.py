import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import tracemalloc
import sys
import tensorflow as tf
import random as rd

def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta

def csv_list(year, month, date):
    dts = [f'{dt.hour}-{dt.minute}.csv' for dt in 
           datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))]
    return dts

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

    elif 'pressure' in path:
        return min_max_scaler(990, 1025, df.values)

def get_param_path(param_name: str, year, month, date):
    if 'rain' in param_name:
        return f'../../../data/rain_image/{year}/{month}/{date}'
    elif 'abs_wind' in param_name:
        return f'../../../data/abs_wind_image/{year}/{month}/{date}'
    elif 'wind' in param_name:
        return f'../../../data/wind_image/{year}/{month}/{date}'
    elif 'temperature' in param_name:
        return f'../../../data/temp_image/{year}/{month}/{date}'
    elif 'humidity' in param_name:
        return f'../../../data/humidity_image/{year}/{month}/{date}'
    elif 'station_pressure' in param_name:
        return f'../../../data/station_pressure_image/{year}/{month}/{date}'
    elif 'seaLevel_pressure' in param_name:
        return f'../../../data/seaLevel_pressure_image/{year}/{month}/{date}'
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


def load_data(dataType='all', params=['rain', 'humidity', 'temperature', 'wind']):

    if dataType == 'all':
        tracemalloc.start()

        time_list = create_time_list()

        input_arr = []
        label_arr = []

        year = 2020
        monthes = ['04', '05', '06', '07', '08', '09', '10']
        for month in monthes:
            log_memory()
            dates = os.listdir(f'../../../data/rain_image/{year}/{month}')
            for date in dates:
                print("... Loading", date)
                params_path = {}
                if len(params) > 0:
                    for par in params:
                        params_path[par] = get_param_path(param_name=par, year=year, month=month, date=date)
                
                for step in range(0, len(time_list) - 6, 6):
                    file_names = [f'{dt.hour}-{dt.minute}.csv' for dt in time_list[step:step+12]]
                    
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
                    label_arr.append(subset_arrs[6])

        input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, len(params_data.keys())])
        label_arr = np.array(label_arr).reshape([len(label_arr), 50, 50, len(params_data.keys())])

        return input_arr, label_arr

    elif dataType == 'selected':
        tracemalloc.start()

        input_arr = []
        label_arr = []

        year = 2020
        monthes = ['04', '05', '06', '07', '08', '09', '10']

        csv_files = csv_list(2020, 1, 1)
        train_list = pd.read_csv('../train_data_list.csv', index_col='date')
        for date in train_list.index:
            log_memory()
            month = date.split('-')[1]

            print("... Loading", date)
            params_path = {}
            if len(params) > 0:
                for par in params:
                    params_path[par] = get_param_path(param_name=par, year=year, month=month, date=date)

            start, end = train_list.loc[date, 'start'], train_list.loc[date, 'end']
            idx_start, idx_end = csv_files.index(start), csv_files.index(end)
            idx_start = idx_start - 12 if idx_start > 11 else 0
            idx_end = idx_end + 12 if idx_end < 132 else 143
            
            for i in range(idx_start, idx_end-12):
                file_names = csv_files[i:i+12]

                subset_arrs = []
                for file_name in file_names:
                    params_data = {}
                    # Load data
                    for par in params:
                        if 'wind' in par and par != 'abs_wind':
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
                label_arr.append(subset_arrs[6])

        input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, len(params_data.keys())])
        label_arr = np.array(label_arr).reshape([len(label_arr), 50, 50, len(params_data.keys())])

        return input_arr, label_arr

def get_data_paths(params=['rain', 'station_pressure', 'seaLevel_pressure']):

    year = 2020

    csv_files = csv_list(2020, 1, 1)
    train_list = pd.read_csv('../train_data_list.csv', index_col='date')

    # paths - date_count - each time id - paths of each parameters
    paths = {}
    for date in train_list.index:
        month = date.split('-')[1]
        params_path = {}
        if len(params) > 0:
            for par in params:
                params_path[par] = get_param_path(param_name=par, year=year, month=month, date=date)

        start, end = train_list.loc[date, 'start'], train_list.loc[date, 'end']
        idx_start, idx_end = csv_files.index(start), csv_files.index(end)
        idx_start = idx_start - 12 if idx_start > 11 else 0
        idx_end = idx_end + 12 if idx_end < 132 else 143
        
        count = 1
        for i in range(idx_start, idx_end-12):
            file_names = csv_files[i:i+12]
            sub_paths = {}
            for file_name in file_names:
                sub_paths[file_name] = {}
                for par in params:
                    if 'wind' in par and par != 'abs_wind':
                        sub_paths[file_name]['u_wind'] = params_path[par] + f'/{file_name}'.replace('.csv', 'U.csv')
                        sub_paths[file_name]['v_wind'] = params_path[par] + f'/{file_name}'.replace('.csv', 'V.csv')
                        
                    else:
                        sub_paths[file_name][par] = params_path[par] + f'/{file_name}'

                paths[f'{date}_{count}'] = sub_paths
                count += 1
    return paths


def load_data_from_paths(paths):
    # paths: dict
    # key0: day and count (e.g. 2021-01-01_55)
    # key1: time_id (12 items e.g. 6-10.csv)
    # key2: parameter names
    input_arr = []
    label_arr = []
    for date_count in paths:
        subset_arrs = []
        for time_id in paths[date_count]:
            params_data = {}
            # Load data
            for par in paths[date_count][time_id].keys():
                params_data[par] = load_csv_data(paths[date_count][time_id][par])

            # Check if data is valid
            for key in params_data.keys():
                if not chack_data_scale(params_data[key]):
                    print(paths[date_count][time_id][par], " Has invalid scale.(x > 1 or x < 0)")
                    sys.exit()
            
            subset_arr = np.empty([50, 50, len(params_data.keys())])
            for i in range(50):
                for j in range(50):
                    for k, key in enumerate(params_data.keys()):
                        subset_arr[i, j, k] = params_data[key][i, j]
                        
            
            subset_arrs.append(subset_arr)
            
        input_arr.append(subset_arrs[:6])
        label_arr.append(subset_arrs[6])

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, len(params_data.keys())])
    label_arr = np.array(label_arr).reshape([len(label_arr), 50, 50, len(params_data.keys())])

    return input_arr, label_arr


def train_data_generator(paths, batch_size=16):
    # get keys
    keys = [*paths.keys()]

    # Custom butch
    file_of_dataset = tf.data.Dataset.from_tensor_slices(keys)
    batched_file = file_of_dataset.batch(batch_size)

    for batched_keys in batched_file.take(batch_size):
        batched_keys = [i.decode('UTF-8') for i in  batched_keys.numpy().tolist()]
        input_arr, label_arr = load_data_from_paths(dict((key, paths[key]) for key in batched_keys))
        yield input_arr, label_arr

def get_train_valid_paths(params=['rain', 'humidity', 'temperature', 'abs_wind', 'seaLevel_pressure']):
    print('[INFO] Load data paths  ...')
    # Load data files
    data_paths = get_data_paths(params)
    # Suffle keys
    keys = [*data_paths.keys()]
    rd.shuffle(keys)

    split_length = len(keys) // 5

    train_paths = dict((key, data_paths[key]) for key in keys[split_length:])
    valid_paths = dict((key, data_paths[key]) for key in keys[:split_length])
    return train_paths, valid_paths

def load_valid_data(paths):
    print('[INFO] Load Validation data ...')
    return load_data_from_paths(paths)

if __name__ == '__main__':
    train_paths, valid_paths = get_train_valid_paths(params=['rain', 'humidity', 'temperature', 'abs_wind', 'seaLevel_pressure'])
    X_valid, y_valid = load_valid_data(valid_paths)
    train_data_generator(train_paths)
