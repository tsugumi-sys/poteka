import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import re

def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta
        
def create_time_list(year, month, date):
    dts = [dt.strftime('%Y-%m-%d T%H:%M Z') for dt in
            datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=2))]
    return dts

def make_dates(x):
    if len(str(x)) == 2:
        return str(x)
    else:
        return "0" + str(x)

folder_path = "../../../data/accumulated-raf-data"


ob_locations = pd.read_csv('../../../p-poteka-config/observation_point.csv', index_col="Name")

years = ['2019']
monthes = ['10', '11']#['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
dates = [make_dates(i) for i in range(1, 32)]

cols = ['LON', 'LAT', 'hour-rain', 'AT1', 'RH1', 'SOL', 'WD1', 'WS1', 'PRS', 'SLP']

data_cols = ['hour-rain', 'AT1', 'RH1', 'SOL', 'WD1', 'WS1', 'PRS', 'SLP']

for year in years:
    for month in monthes:
        for date in dates:
            count = 0
            for ob_point in os.listdir(folder_path):
                path = folder_path + f'/{ob_point}/{year}/{month}/{year}-{month}-{date}/data.csv'
                if os.path.exists(path):
                    count += 1
            
            # Create One day Data
            try:
                time_list = create_time_list(int(year), int(month), int(date))
                print(f'{year}-{month}-{date} ', count)
                ob_data = []
                ob_names = []
                ob_lon_lat = []
                for ob_point in os.listdir(folder_path):
                    path = folder_path + f'/{ob_point}/{year}/{month}/{year}-{month}-{date}/data.csv'
                    if os.path.exists(path):
                        ob_names.append(ob_point)
                        ob_df = pd.read_csv(path, index_col='Datetime')
                        ob_data.append(ob_df)
                        ob_lon, ob_lat = ob_locations.loc[ob_point, 'LON'], ob_locations.loc[ob_point, 'LAT']
                        ob_lon_lat.append([ob_lon, ob_lat])
                
                
                # Save Folder
                save_folder_path = "../../../data/one_day_data"
                create_folder = [year, month, f'{year}-{month}-{date}']
                for folder_name in create_folder:
                    save_folder_path = save_folder_path + f'/{folder_name}'
                    if not os.path.exists(save_folder_path):
                        os.mkdir(save_folder_path)
                
                for time in time_list:
                    df = pd.DataFrame(index=ob_names, columns=cols)
                    datetime_obj = datetime.strptime(time, '%Y-%m-%d T%H:%M Z')
                    #print(time, datetime_obj.hour, datetime_obj.minute)
                    for i in range(len(ob_data)):
                        data = ob_data[i]
                        ob_name = ob_names[i]
                        ob_point = ob_lon_lat[i]
                        df.loc[ob_name, 'LON'], df.loc[ob_name, 'LAT'] = ob_point
                        for col in data_cols:
                            df.loc[ob_name, col] = data.loc[time, col]
                    #print(df)
                    
                    save_path = save_folder_path + f'/{datetime_obj.hour}-{datetime_obj.minute}.csv'
                    df.to_csv(save_path)
            except ValueError:
                print(f'{year}-{month}-{date} does not exists')
                continue