import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import tracemalloc

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
        # abs_wind [0, 30]
        return min_max_scaler(-10, 10, df.values)

def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta
        
def create_time_list(year=2020, month=1, date=1):
    dts = [dt for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))]
    return dts


def load_data_RUV(dataType='all'): # Rain U-Wind V-Wind

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
                print(date)
                rain_path = f'../../../data/rain_image/{year}/{month}/{date}'
                wind_path = f'../../../data/wind_image/{year}/{month}/{date}'
                if os.path.exists(rain_path) and os.path.exists(wind_path):
                    rain_file_num = len(os.listdir(rain_path))
                    wind_file_num = len(os.listdir(wind_path))
                    if rain_file_num + wind_file_num == 288 * 3:
                        for step in range(0, len(time_list) - 6, 6):
                            file_names = [f'{dt.hour}-{dt.minute}.csv' for dt in time_list[step:step+12]]
                            
                            subset_arrs = []
                            for file_name in file_names:
                                # Load data
                                rain_file_path = rain_path + f'/{file_name}'
                                u_wind_file_path = wind_path + f'/{file_name}'.replace('.csv', 'U.csv')
                                v_wind_file_path = wind_path + f'/{file_name}'.replace('.csv', 'V.csv')
                                
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
                            label_arrs = []
                            for i in range(6):
                                arr = np.empty([50, 50])
                                for j in range(50):
                                    for k in range(50):
                                        arr[j, k] = subset_arrs[6:][i][j, k, 0]
                                label_arrs.append(arr)
                            label_arr.append(label_arrs)


        input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
        label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 1])

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
            print(date)

            rain_path = f'../../../data/rain_image/{year}/{month}/{date}'
            wind_path = f'../../../data/wind_image/{year}/{month}/{date}'
            if os.path.exists(rain_path) and os.path.exists(wind_path):
                start, end = train_list.loc[date, 'start'], train_list.loc[date, 'end']
                idx_start, idx_end = csv_files.index(start), csv_files.index(end)
                idx_start = idx_start - 12 if idx_start > 11 else 0
                idx_end = idx_end + 12 if idx_end < 132 else 143
                for i in range(idx_start, idx_end-12):
                    file_names = csv_files[i:i+12]
                    
                    subset_arrs = []
                    for file_name in file_names:
                        # Load data
                        rain_file_path = rain_path + f'/{file_name}'
                        u_wind_file_path = wind_path + f'/{file_name}'.replace('.csv', 'U.csv')
                        v_wind_file_path = wind_path + f'/{file_name}'.replace('.csv', 'V.csv')
                        
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
                    label_arrs = []
                    for i in range(6):
                        arr = np.empty([50, 50])
                        for j in range(50):
                            for k in range(50):
                                arr[j, k] = subset_arrs[6:][i][j, k, 0]
                        label_arrs.append(arr)
                    label_arr.append(label_arrs)

                    # Check Nan
                    for arr in subset_arrs:
                        nan_count = np.count_nonzero(np.isnan(arr))
                        if nan_count > 0:
                            print("NaN here: ", date)
        input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
        label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 1])

        return input_arr, label_arr

def load_data_RWT(): # Rain Absolute Wind Temperature
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
            print(date)
            rain_path = f'../../../data/rain_image/{year}/{month}/{date}'
            temp_path = f'../../../data/temp_image/{year}/{month}/{date}'
            abs_wind_path = f'../../../data/abs_wind_image/{year}/{month}/{date}'
            if os.path.exists(rain_path) and os.path.exists(abs_wind_path) and os.path.exists(temp_path):
                rain_file_num = len(os.listdir(rain_path))
                temp_file_num = len(os.listdir(temp_path))
                abs_wind_file_num = len(os.listdir(abs_wind_path))
                if rain_file_num == 288:
                    for step in range(0, len(time_list) - 6, 6):
                        file_names = [f'{dt.hour}-{dt.minute}.csv' for dt in time_list[step:step+12]]
                        
                        subset_arrs = []
                        for file_name in file_names:
                            # Load data
                            rain_file_path = rain_path + f'/{file_name}'
                            temp_file_path = temp_path + f'/{file_name}'
                            abs_wind_file_path = abs_wind_path + f'/{file_name}'
                            
                            # Create ndarray
                            rain_arr = load_csv_data(rain_file_path)
                            temp_arr = load_csv_data(temp_file_path)
                            abs_wind_arr = load_csv_data(abs_wind_file_path)
                            
                            #print('Absolute Wind Data', abs_wind_arr.max(), abs_wind_arr.min())
                            
                            subset_arr = np.empty([50, 50, 3])
                            for i in range(50):
                                for j in range(50):
                                    subset_arr[i, j, 0] = rain_arr[i, j]
                                    subset_arr[i, j, 1] = temp_arr[i, j]
                                    subset_arr[i, j, 2] = abs_wind_arr[i, j]
                            
                            subset_arrs.append(subset_arr)
                        
                        input_arr.append(subset_arrs[:6])
                        label_arr.append(subset_arrs[6:])

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
    label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 3])

    return input_arr, label_arr


def load_rain_data():
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
            print(date)
            rain_path = f'../../../data/rain_image/{year}/{month}/{date}'
            if os.path.exists(rain_path):
                rain_file_num = len(os.listdir(rain_path))
                if rain_file_num == 288:
                    for step in range(0, len(time_list) - 6, 6):
                        file_names = [f'{dt.hour}-{dt.minute}.csv' for dt in time_list[step:step+12]]
                        
                        subset_arrs = []
                        for file_name in file_names:
                            # Load data
                            rain_file_path = rain_path + f'/{file_name}'
                            
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

    input_arr = np.array(input_arr).reshape([len(input_arr), 6, 50, 50, 3])
    label_arr = np.array(label_arr).reshape([len(label_arr), 6, 50, 50, 3])

    return input_arr, label_arr

if __name__ == '__main__':
    input_arr, label_arr = load_data_RUV(dataType='selected')
    print(input_arr.shape, label_arr.shape)