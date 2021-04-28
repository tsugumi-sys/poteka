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

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)


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
                                grid_lon = np.round(np.linspace(120.90, 121.150, 75), decimals=3)
                                grid_lat = np.round(np.linspace(14.350, 14.760, 75), decimals=3)
                                xi, yi = np.meshgrid(grid_lon, grid_lat)
                                z1 = rbfi(xi, yi)
                                rain_data = np.where(z1 > 0, z1, 0)
                                fig = plt.figure(figsize=(7, 8), dpi=80)
                                ax = plt.axes(projection=ccrs.PlateCarree())
                                ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                ax.add_feature(cfeature.COASTLINE)

                                clevs = [0, 5, 7.5, 10, 15, 20, 30, 40,
                                        50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750]
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
                                            (0.125490203499794, 0.125490203499794, 0.501960813999176),
                                            (0.125490203499794, 0.125490203499794, 0.125490203499794),
                                            (0.501960813999176, 0.501960813999176, 0.501960813999176),
                                            (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
                                            (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
                                            (0.8549019694328308, 0.6509804129600525, 0.47058823704719543)]
                                cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
                                norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                cs = ax.contourf(xi, yi, rain_data, clevs, cmap=cmap, norm=norm)
                                cbar = plt.colorbar(cs, orientation='vertical')
                                cbar.set_label('millimeter')
                                ax.scatter(df['LON'], df['LAT'], marker='D', color='dimgrey')
                                
                                # Save Image
                                save_path = '../../../data/rain_image'
                                folders = [year, month, date]
                                for folder in folders:
                                    if not os.path.exists(save_path + f'/{folder}'):
                                        os.mkdir(save_path + f'/{folder}')
                                    save_path += f'/{folder}'
                                save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                plt.savefig(save_path)
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
                        