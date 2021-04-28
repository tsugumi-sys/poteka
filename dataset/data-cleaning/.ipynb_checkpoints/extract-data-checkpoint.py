import os
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta

# Create the list of the every 1 minute datetime of a day
# Parameter
# ======================
# start: datetime of start of the day e.g. datetime(2020, 7, 21, 0)
# end: datetime of the end of the day e.g. datetime(2020, 7, 21, 23, 59)
# delta: integer how many times steps to next index e.g. 1 
def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta

# Create Formated datetime list
# Parameter
# ========================
# year: the number of the year
# month: ...
# date: ...
def create_time_list(year, month, date):
    dts = [dt.strftime('%Y-%m-%d T%H:%M Z') for dt in 
            datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=1))]
    return dts

# Find weather values from 'RAF 0000' -> 0.0
# Parameter
# =========================
# x: the value of cells of csv file
def find_value(x):
    result = re.findall('\d+', x)
    if len(result) > 1:
        return result[-1]
    elif len(result) == 1:
        return result[0]
    else:
        return np.nan

# Format 14 digir number to datetime
# Parameter
# ==========================
# x: the value of cells of csv file
def format_datetime(x):
    x = str(x)
    year = int(x[:4])
    month = int(x[4:6])
    date = int(x[6:8])
    hour = int(x[8:10])
    minute = int(x[10:12])
    dt = datetime(year, month, date, hour, minute)
    return dt.strftime('%Y-%m-%d T%H:%M Z')



cols = ['Datetime', 'RAF', 'RA1', 'RI1', 'ERA', 'CRA', 'AT1', 'RH1', 'PRS', 'SLP', 'WD1', 'WDM', 'WS1', 
        'WSM', 'WND', 'WNS', 'SOL', 'WET', 'WBG', 'WEA']


# magnification of weather values
magni = [10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 10, 10, 1, 10, 1, 1, 10, 1]

root_path = '../../../data/poteka-raw-data/'

years = ['2019', '2020']
monthes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
for observation_point in os.listdir(root_path):
    for year in years:
        for month in monthes:
            path = root_path + observation_point + f'/{year}/{month}/'
            # Check if the observation point has the data of year/month
            if os.path.exists(path):
                for date in os.listdir(path):
                    csv_path = path + f'{date}/Weather/'
                    #  Check if the observation point has the data of year/month/date and csv file of the data
                    if os.path.exists(csv_path) and len(os.listdir(csv_path)) > 0:
                        filename = os.listdir(csv_path)[0]
                        df_before = pd.read_csv(csv_path + filename, header=None)

                        # Extract number of the data from the cell's value
                        for col in cols:
                            for col_def in df_before.columns:
                                if col in str(df_before[col_def].values[0]):
                                    df_before[col] = df_before[col_def].apply(find_value)
                                    df_before[col] = df_before[col].apply(float)

                        # Extract datetime and format correctlly
                        for col in df_before.columns:
                            val = df_before[col].values[0]
                            check = find_value(str(val))
                            if len(check) == 14:
                                y = int(str(val)[:4])
                                m = int(str(val)[4:6])
                                d = int(str(val)[6:8])
                                df_before['Datetime'] = df_before[col].apply(str)
                                df_before['Datetime'] = df_before['Datetime'].apply(format_datetime)

                        df_before = df_before[cols]
                        df_before = df_before.set_index('Datetime')

                        # Rescale data
                        for i in range(len(cols[1:])):
                            col = cols[i+1]
                            mag = magni[i]
                            print(col, mag)
                            df_before[col] = df_before[col] / mag
                        
                        # Reset index
                        time_list = create_time_list(y, m, d)
                        df = df_before.reindex(time_list)

                        save_path = '../../../data/cleaned-data/'
                        oymd = [observation_point, year, month, date]
                        for item in oymd:
                            save_path += str(item)
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                        
                        df.to_csv(save_path + 'data.csv')
                        
