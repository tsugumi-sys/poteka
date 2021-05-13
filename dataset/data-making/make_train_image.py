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
from scipy.interpolate import Rbf
import gstools as gs
import random
import requests
import tracemalloc
from PIL import Image
from matplotlib import cm

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

# Rainbow Color
def make_rain_image():
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
                                print('PATH: ', path)
                                
                                try:
                                    df = pd.read_csv(path, index_col=0)
                                    df['hour-rain'] = np.where(df['hour-rain'] > 0, df['hour-rain'], round(random.uniform(0.1, 0.8), 5))
                                    rbfi = Rbf(df['LON'], df['LAT'], df['hour-rain'], function='gaussian')
                                    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
                                    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
                                    xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    z1 = rbfi(xi, yi)
                                    rain_data = np.where(z1 > 0, z1, 0)
                                    rain_data = np.where(rain_data > 150, 150, rain_data)
                                    fig = plt.figure(figsize=(7, 8), dpi=80)
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False

                                    clevs = [0, 5, 7.5, 10, 15, 20, 30, 40,
                                            50, 70, 100, 150]
                                    cmap_data = [(1.0, 1.0, 1.0),
                                                (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
                                                (0.0, 1.0, 1.0),
                                                (0.0, 0.8784313797950745, 0.501960813999176),
                                                (0.0, 0.7529411911964417, 0.0),
                                                (0.501960813999176, 0.8784313797950745, 0.0),
                                                (1.0, 1.0, 0.0),
                                                (1.0, 0.6274510025978088, 0.0),
                                                (1.0, 0.0, 0.0),
                                                (1.0, 0.125490203499794, 0.501960813999176),
                                                (0.9411764740943909, 0.250980406999588, 1.0),
                                                (0.501960813999176, 0.125490203499794, 1.0),
                                                (0.250980406999588, 0.250980406999588, 1.0),
                                                ]
                                    cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(xi, yi, rain_data, clevs, cmap=cmap, norm=norm)
                                    cbar = plt.colorbar(cs, orientation='vertical')
                                    cbar.set_label('millimeter')
                                    
                                    # Save Image and Csv
                                    save_path = '../../../data/train_with_image/rain_image'
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f'/{folder}'):
                                            os.mkdir(save_path + f'/{folder}')
                                        save_path += f'/{folder}'
                                    save_csv_path = save_path + f'/{data_file}'
                                    save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                    plt.savefig(save_path)

                                    im = Image.open(save_path)
                                    im.crop((118, 77, 417, 570)).save(save_path.replace('.png', 'croped.png'), quality=100)

                                    print('Sucessfully Saved')

                                    plt.close()
                                except:
                                    print('!'*10,' Failed ', '!'*10)
                                    failed_path.append(path)
                                    continue
        failed = pd.DataFrame({'path': failed_path})
        failed.to_csv('failed.csv')
        send_line_notify('Succeccfuly Completed!!!')
    except:
        import traceback
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())

# Gradation Color
def make_dense_rain_image():
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
                                print('PATH: ', path)
                                
                                try:
                                    df = pd.read_csv(path, index_col=0)
                                    df['hour-rain'] = np.where(df['hour-rain'] > 0, df['hour-rain'], round(random.uniform(0.1, 0.8), 5))
                                    rbfi = Rbf(df['LON'], df['LAT'], df['hour-rain'], function='gaussian')
                                    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
                                    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
                                    xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    z1 = rbfi(xi, yi)
                                    rain_data = np.where(z1 > 0, z1, 0)
                                    rain_data = np.where(rain_data > 150, 150, rain_data)
                                    fig = plt.figure(figsize=(7, 8), dpi=80)
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False

                                    clevs = [i for i in range(1, 151)]
                                    

                                    cmap = cm.BuPu
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(xi, yi, rain_data, clevs, cmap=cmap, norm=norm)
                                    cbar = plt.colorbar(cs, orientation='vertical')
                                    cbar.set_label('millimeter')
                                    
                                    # Save Image and Csv
                                    save_path = '../../../data/train_with_image/dense_rain_image'
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f'/{folder}'):
                                            os.mkdir(save_path + f'/{folder}')
                                        save_path += f'/{folder}'
                                    save_csv_path = save_path + f'/{data_file}'
                                    save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                    plt.savefig(save_path)

                                    im = Image.open(save_path)
                                    im.crop((118, 77, 417, 570)).save(save_path.replace('.png', 'croped.png'), quality=100)

                                    print('Sucessfully Saved')

                                    plt.close()
                                except:
                                    print('!'*10,' Failed ', '!'*10)
                                    failed_path.append(path)
                                    continue
        failed = pd.DataFrame({'path': failed_path})
        failed.to_csv('failed.csv')
        send_line_notify('Succeccfuly Completed!!!')
    except:
        import traceback
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())
                        
if __name__ == '__main__':
    make_dense_rain_image()