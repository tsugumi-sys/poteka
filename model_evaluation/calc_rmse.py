import pandas as pd
import os
from datetime import datetime, timedelta

from pandas.core.algorithms import mode

def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta

def csv_list(year, month, date):
    dts = [f'{dt.hour}-{dt.minute}.csv' for dt in 
           datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))]
    return dts

# Calculate RMSE of whole period
def calc_period_rmse(model_type, model_name, time_span):
    # load validation date and time
    valid_list = pd.read_csv('valid_data_list.csv')

    # Create csv list
    csv_files = csv_list(2020, 1, 1)

    root = f'../../data/prediction_image/{model_type}/{time_span}min_{model_name}/2019/'
    rmses = []
    dates = valid_list['date'].unique()
    for date in dates:
        
        month = date.split('-')[1]
        rmse_df = pd.read_csv(root + month + '/RMSE/' + date + '.csv')

        files = valid_list.loc[valid_list['date'] == date]
        for i in files.index:
            start, end = valid_list.loc[i, 'start'], valid_list.loc[i, 'end']
            idx_start, idx_end = csv_files.index(start), csv_files.index(end)
            rmses.append(rmse_df[idx_start:idx_end+1]['RMSE'].mean())
        
    print('='*60)
    print(model_type, model_name, time_span, 'RMSE', sum(rmses) / len(rmses))


# Calculate RMSE of each time and day
def calc_rmse(model_type='oneByone_model', model_name='model1', time_span=60):
    print('-'*60)
    print(model_type, model_name, time_span)
    print('-'*60)
    csv_files = csv_list(2020,1,1)
    monthes = ['10', '11']
    prediction_root = f'../../data/prediction_image/{model_type}/{time_span}min_{model_name}/2019/'
    label_root = '../../data/rain_image/2019/'

    for month in monthes:
        if not os.path.exists(prediction_root + month + '/RMSE/'):
            os.mkdir(prediction_root + month + '/RMSE/')

        for date in os.listdir(label_root + month):
            rmse_df = pd.DataFrame()
            print('Calculating RMSE ...', date)
            for csv_file in csv_files[6:]:
                label_data = pd.read_csv(label_root + month + f'/{date}/{csv_file}', index_col=0)
                pred_data = pd.read_csv(prediction_root + month + f'/{date}/{csv_file}', index_col=0)
                rmse = ((label_data.values - pred_data.values) ** 2).mean() ** .5
                rmse_df.loc[csv_file, 'RMSE'] = rmse

            rmse_df.to_csv(prediction_root + month + f'/RMSE/{date}.csv')

if __name__ == '__main__':
    # calc_rmse(model_type='oneByone_model', model_name='ruvth_baseline', time_span=60)
    # calc_rmse(model_type='oneByone_model', model_name='ruvth_optuned', time_span=60)

    models = {
        'oneByone_model': {
            'model_names': ['ruvth_baseline', 'ruvth_optuned'],
            'time_span': 60,
        }
    }
    
    for model_type in models.keys():
        time_span = models['oneByone_model']['time_span']
        for model_name in models['oneByone_model']['model_names']:
            calc_period_rmse(model_type=model_type, model_name=model_name, time_span=time_span)