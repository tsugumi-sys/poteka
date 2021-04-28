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
    result = re.findall('\d+', str(x))
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

# Find correct format csv file of the data
# Parameter
# ==========================
# path: {string} the path of the directory that the csv file is in it
# output: {string} the name of the csv file
def find_csv_file(path):
    files = os.listdir(path)
    result = []
    for file in files:
        for item in re.findall('^weather.+.csv$', str(file)):
            result.append(item)
    if len(result) > 1:
        return result[-1]
    else:
        return result[0]
    return result



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
                        print('-'*50)
                        print(csv_path)
                        filename = find_csv_file(csv_path)
                        df_before = pd.read_csv(csv_path + filename, header=None, error_bad_lines=False)
                        print(df_before.info())

                        # Extract number of the data from the cell's value
                        for col in cols:
                            for col_def in df_before.columns:
                                if col in str(df_before[col_def].values[0]):
                                    df_before[col] = df_before[col_def].apply(find_value)
                                    df_before[col] = df_before[col].apply(float)
                        df_before.dropna(inplace=True)

                        if df_before.values.size > 0:
                            # Extract datetime and format correctlly
                            for col in df_before.columns:
                                val = df_before[col].values[0]
                                check = find_value(str(val).replace('.0', ''))
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
                                df_before[col] = df_before[col] / mag
                            
                            # Reset index if there is a lack of the data
                            time_list = create_time_list(y, m, d)
                            df_before = df_before.drop_duplicates()
                            print(len(time_list), len(df_before.index))
                            print(df_before.info())
                            df = df_before.reindex(time_list)

                            save_path = '../../../data/cleaned-data/'
                            oymd = [observation_point, year, month, date]
                            for item in oymd:
                                save_path += f'{str(item)}/'
                                if not os.path.exists(save_path):
                                    os.mkdir(save_path)
                            
                            
                            df.to_csv(save_path + 'data.csv')
                        else:
                            continue
                        
