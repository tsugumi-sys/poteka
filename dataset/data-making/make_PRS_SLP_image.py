import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
from pykrige.kriging_tools import write_asc_grid
from scipy.interpolate import Rbf, RBFInterpolator
import gstools as gs
import random
import requests
from matplotlib import cm
import tracemalloc
from dotenv import load_dotenv
from pathlib import Path
import traceback

dotenv_path = Path('../../.env')
load_dotenv(dotenv_path=dotenv_path)

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = os.getenv('LINE_TOKEN')
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

# PRS: station pressure
def make_prs_image():
    tracemalloc.start()
    failed_path = []
    try:
        root_folder = '../../../data/one_day_data'

        for year in os.listdir(root_folder):
            for month in os.listdir(root_folder + f'/{year}'):
                for date in os.listdir(root_folder + f'/{year}/{month}'):
                    if len(os.listdir(root_folder + f'/{year}/{month}/{date}')) > 0:
                        data_files = os.listdir(root_folder + f'/{year}/{month}/{date}')
                        for data_file in data_files:
                            path = root_folder + f'/{year}/{month}/{date}/{data_file}'
                            if os.path.exists(path):
                                print('-'*80)
                                print('Station Pressure')
                                print('PATH: ', path)
                                try:
                                    df = pd.read_csv(path, index_col=0)
                                    rbfi = RBFInterpolator(df[['LON', 'LAT']], df['PRS'], kernel='linear', epsilon=10)
                                    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
                                    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
                                    # xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    xgrid = np.around(np.mgrid[120.90:121.150:50j, 14.350:14.760:50j], decimals=3)
                                    xfloat = xgrid.reshape(2, -1).T

                                    z1 = rbfi(xfloat)
                                    z1 = z1.reshape(50,50)
                                    
                                    humid_data = np.where(z1 > 990, z1, 990)
                                    humid_data = np.where(humid_data > 1025, 1025, humid_data)
                                    fig = plt.figure(figsize=(7, 8), dpi=80)
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    ax.add_feature(cfeature.COASTLINE)
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False

                                    clevs = [i for i in range(990, 1026, 1)]
                                    
                                    cmap = cm.jet
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(*xgrid, humid_data, clevs, cmap=cmap, norm=norm)
                                    cbar = plt.colorbar(cs, orientation='vertical')
                                    cbar.set_label('hPa')
                                    ax.scatter(df['LON'], df['LAT'], marker='D', color='dimgrey')
                                    for i, val in enumerate(df['PRS']):
                                        ax.annotate(val, (df['LON'][i], df['LAT'][i]))
                                    ax.set_title('Station Pressure')
                                    
                                    # Save Image and Csv
                                    save_path = '../../../data/station_pressure_image'
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f'/{folder}'):
                                            os.mkdir(save_path + f'/{folder}')
                                        save_path += f'/{folder}'
                                    save_csv_path = save_path + f'/{data_file}'
                                    save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                    plt.savefig(save_path)

                                    save_df = pd.DataFrame(humid_data)
                                    save_df = save_df[save_df.columns[::-1]].T
                                    save_df.columns = grid_lon
                                    save_df.index = grid_lat[::-1]
                                    save_df.to_csv(save_csv_path)
                                    print('Sucessfully Saved')

                                    plt.close()
                                except:
                                    print('!'*10,' Failed ', '!'*10)
                                    print(traceback.format_exc())
                                    failed_path.append(path)
                                    continue
        failed = pd.DataFrame({'path': failed_path})
        failed.to_csv('failed.csv')
        send_line_notify('Succeccfuly Completed!!!')
    except:
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())

# SLP: sea level pressure
def make_slp_image():
    tracemalloc.start()
    failed_path = []
    try:
        root_folder = '../../../data/one_day_data'

        for year in os.listdir(root_folder):
            for month in os.listdir(root_folder + f'/{year}'):
                for date in os.listdir(root_folder + f'/{year}/{month}'):
                    if len(os.listdir(root_folder + f'/{year}/{month}/{date}')) > 0:
                        data_files = os.listdir(root_folder + f'/{year}/{month}/{date}')
                        for data_file in data_files:
                            path = root_folder + f'/{year}/{month}/{date}/{data_file}'
                            if os.path.exists(path):
                                print('-'*80)
                                print('Sea Level Pressure')
                                print('PATH: ', path)
                                try:
                                    df = pd.read_csv(path, index_col=0)
                                    rbfi = RBFInterpolator(df[['LON', 'LAT']], df['SLP'], kernel='linear', epsilon=10)
                                    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
                                    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
                                    # xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    xgrid = np.around(np.mgrid[120.90:121.150:50j, 14.350:14.760:50j], decimals=3)
                                    xfloat = xgrid.reshape(2, -1).T

                                    z1 = rbfi(xfloat)
                                    z1 = z1.reshape(50,50)
                                    
                                    humid_data = np.where(z1 > 990, z1, 990)
                                    humid_data = np.where(humid_data > 1025, 1025, humid_data)
                                    fig = plt.figure(figsize=(7, 8), dpi=80)
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    ax.add_feature(cfeature.COASTLINE)
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False

                                    clevs = [i for i in range(990, 1026, 1)]
                                    
                                    cmap = cm.jet
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(*xgrid, humid_data, clevs, cmap=cmap, norm=norm)
                                    cbar = plt.colorbar(cs, orientation='vertical')
                                    cbar.set_label('hPa')
                                    ax.scatter(df['LON'], df['LAT'], marker='D', color='dimgrey')
                                    for i, val in enumerate(df['SLP']):
                                        ax.annotate(val, (df['LON'][i], df['LAT'][i]))
                                    ax.set_title('Sea Level Pressure')
                                    
                                    # Save Image and Csv
                                    save_path = '../../../data/seaLevel_pressure_image'
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f'/{folder}'):
                                            os.mkdir(save_path + f'/{folder}')
                                        save_path += f'/{folder}'
                                    save_csv_path = save_path + f'/{data_file}'
                                    save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                    plt.savefig(save_path)

                                    save_df = pd.DataFrame(humid_data)
                                    save_df = save_df[save_df.columns[::-1]].T
                                    save_df.columns = grid_lon
                                    save_df.index = grid_lat[::-1]
                                    save_df.to_csv(save_csv_path)
                                    print('Sucessfully Saved')

                                    plt.close()
                                except:
                                    print('!'*10,' Failed ', '!'*10)
                                    print(traceback.format_exc())
                                    failed_path.append(path)
                                    continue
        failed = pd.DataFrame({'path': failed_path})
        failed.to_csv('failed.csv')
        send_line_notify('Succeccfuly Completed!!!')
    except:
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())
                        
if __name__ == '__main__':
    make_prs_image()
    #make_slp_image()