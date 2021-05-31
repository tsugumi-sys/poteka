import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import Rbf
from matplotlib import cm
from metpy.units import units
from metpy.calc import wind_components, divergence, lat_lon_grid_deltas
from metpy.interpolate import interpolate_to_grid, remove_nan_observations
import requests
import tracemalloc
import traceback

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

def calc_u_v(df, ob_point):
    wind_dir = df['WD1']
    wind_speed = df['WS1']
    wind_u_v = wind_components(wind_speed * units('m/s'), wind_dir * units.deg)
    return [ob_point, round(wind_u_v[0].magnitude, 5), round(wind_u_v[1].magnitude, 5)] # (index, u wind, v wind) u: X (East-West) v: Y(North-South)
 

def make_wind_image():

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
                                    wind_df = pd.DataFrame([ calc_u_v(df.loc[i, :], i) for i in df.index ], columns=["OB-POINT", "U-WIND", "V-WIND"])
                                    wind_df = wind_df.set_index("OB-POINT")
                                    wind_df['LON'] = df['LON']
                                    wind_df['LAT'] = df['LAT']
                                    grid_size = 50
                                    v_wind_rbfi = Rbf(wind_df['LON'], wind_df['LAT'], wind_df['V-WIND'].values, function='gaussian')
                                    u_wind_rbfi = Rbf(wind_df['LON'], wind_df['LAT'], wind_df['U-WIND'].values, function='gaussian')
                                    grid_lon = np.round(np.linspace(120.90, 121.150, grid_size), decimals=3)
                                    grid_lat = np.round(np.linspace(14.350, 14.760, grid_size), decimals=3)
                                    xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    v_wind = v_wind_rbfi(xi, yi)
                                    u_wind = u_wind_rbfi(xi, yi)
                                    v_wind = np.where(v_wind > 20, 20, v_wind)
                                    v_wind = np.where(v_wind < -20, -20, v_wind)
                                    u_wind = np.where(u_wind > 20, 20, u_wind)
                                    u_wind = np.where(u_wind < -20, -20, u_wind)
                                    
                                    v_wind_grad = np.array(np.gradient(v_wind)[1])
                                    u_wind_grad = np.array(np.gradient(u_wind)[0])
                                    wind_div = np.empty([grid_size, grid_size])
                                    for i in range(grid_size):
                                        for j in range(grid_size):
                                            x_left = i - 1 if i - 1 > 0 else 0
                                            x_right = i + 1 if i + 1 < grid_size - 1 else grid_size - 1
                                            y_above = j + 1 if j + 1 < grid_size - 1 else grid_size - 1
                                            y_bottom = j - 1 if j - 1 > 0 else 0
                                            val = 0
                                            for x in range(x_left, x_right + 1):
                                                for y in range(y_bottom, y_above + 1):
                                                    val += v_wind_grad[x, y] + u_wind_grad[x, y]
                                            wind_div[i, j] = val
                                    
                                    wind_div = np.where(wind_div > 10, 10, wind_div)
                                    wind_div = np.where(wind_div < -10, -10, wind_div)
                                    print(wind_div.max(), wind_div.min())

                                    # Save Fig
                                    fig = plt.figure(figsize=(7, 8), dpi=80)
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    ax.add_feature(cfeature.COASTLINE)
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False
                                    
                                    # Colror Bar
                                    clevs = list(range(-10, 11))
                                    cmap = cm.coolwarm
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(xi, yi, wind_div, clevs, cmap=cmap, norm=norm)
                                    cbar = plt.colorbar(cs, orientation='vertical')
                                    cbar.set_label('wind divergence (/second)')
                                    plt.quiver(xi, yi, u_wind, v_wind)
                                    ax.scatter(df['LON'], df['LAT'], marker='D', color='dimgrey')

                                    # Save Image and CSV
                                    save_path = '../../../data/wind_image'
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f'/{folder}'):
                                            os.mkdir(save_path + f'/{folder}')
                                        save_path += f'/{folder}'
                                    save_csv_path = save_path + f'/{data_file}'
                                    save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                    
                                    plt.savefig(save_path)

                                    save_df = pd.DataFrame(wind_div, index=np.flip(grid_lat), columns=grid_lon)
                                    save_df.to_csv(save_csv_path)
                                    print('Sucessfully Saved')

                                    plt.close()
                                except:
                                    print('!'*10,' Failed ', '!'*10)
                                    failed_path.append(path)
                                    print(traceback.format_exc())
                                    continue
        failed = pd.DataFrame({'path': failed_path})
        failed.to_csv('failed.csv')
        send_line_notify('Succeccfuly Completed!!!')
    except:
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())


def make_abs_wind_image():

    tracemalloc.start()
    failed_path = []
    try:
        root_folder = '../../../data/one_day_data'

        for year in ['2019']:#os.listdir(root_folder):
            for month in ['10', '11']:#os.listdir(root_folder + f'/{year}'):
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
                                    
                                    # Interpolate Data
                                    grid_size = 50
                                    wind_rbfi = Rbf(df['LON'], df['LAT'], df['WS1'].values, function='gaussian')
                                    
                                    grid_lon = np.round(np.linspace(120.90, 121.150, grid_size), decimals=3)
                                    grid_lat = np.round(np.linspace(14.350, 14.760, grid_size), decimals=3)
                                    xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    
                                    abs_wind = wind_rbfi(xi, yi)
                                    abs_wind = np.where(abs_wind > 30, 30, abs_wind)
                                    abs_wind = np.where(abs_wind < 0, 0, abs_wind)
                                    

                                    # Save Fig
                                    fig = plt.figure(figsize=(7, 8), dpi=80)
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    ax.add_feature(cfeature.COASTLINE)
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False
                                    
                                    #clevs = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                                    clevs = list(range(0, 31, 2))
                                    cmap = cm.viridis
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(xi, yi, abs_wind, clevs, cmap=cmap, norm=norm)
                                    cbar = plt.colorbar(cs, orientation='vertical')
                                    cbar.set_label('wind speed (meter/second)')
                                    ax.scatter(df['LON'], df['LAT'], marker='D', color='dimgrey')

                                    # Save Image and CSV
                                    save_path = '../../../data/abs_wind_image'
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f'/{folder}'):
                                            os.mkdir(save_path + f'/{folder}')
                                        save_path += f'/{folder}'
                                    save_csv_path = save_path + f'/{data_file}'
                                    save_path += '/{}'.format(data_file.replace('.csv', '.png'))
                                    
                                    plt.savefig(save_path)

                                    save_df = pd.DataFrame(abs_wind, index=np.flip(grid_lat), columns=grid_lon)
                                    save_df.to_csv(save_csv_path)
                                    print('Sucessfully Saved')

                                    plt.close()
                                except:
                                    print('!'*10,' Failed ', '!'*10)
                                    failed_path.append(path)
                                    print(traceback.format_exc())
                                    continue
        failed = pd.DataFrame({'path': failed_path})
        failed.to_csv('failed.csv')
        send_line_notify('Succeccfuly Completed!!!')
    except:
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())
        
if __name__ == '__main__':
    #make_wind_image()
    make_abs_wind_image()
                        