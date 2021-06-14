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

def make_rain_image(path: str):
    original_df = pd.read_csv('../../p-poteka-config/observation_point.csv', index_col='Name')
    data = pd.read_csv(path, index_col=0)
    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
    grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
    xi, yi = np.meshgrid(grid_lon, grid_lat)
    fig = plt.figure(figsize=(7, 8), dpi=80)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([120.90, 121.150, 14.350, 14.760])
    ax.add_feature(cfeature.COASTLINE)
    gl = ax.gridlines(draw_labels=True, alpha=0)
    gl.right_labels = False
    gl.top_labels = False

    clevs = [0, 5, 7.5, 10, 15, 20, 30, 40,
            50, 70, 100]
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
                ]
    cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
    norm = mcolors.BoundaryNorm(clevs, cmap.N)

    cs = ax.contourf(xi, yi, data, clevs, cmap=cmap, norm=norm)
    cbar = plt.colorbar(cs, orientation='vertical')
    cbar.set_label('millimeter')
    ax.scatter(original_df['LON'], original_df['LAT'], marker='D', color='dimgrey')

    save_path = path.replace('.csv', '.png')
    plt.savefig(save_path)
    plt.close()
    print(save_path)
    print('Sucessfully Saved')


def make_prediction_image(model_name='model1', time_span=60):
    root_path = f'../../data/prediction_image/{time_span}min_{model_name}'
    for year in os.listdir(root_path):
        for month in os.listdir(root_path + f'/{year}'):
            for date in os.listdir(root_path + f'/{year}/{month}'):
                path = root_path + f'/{year}/{month}/{date}/'
                csv_files = [file_name for file_name in os.listdir(path) if '.csv' in file_name]
                print(csv_files)
                for file_name in csv_files:
                    if os.path.exists(path + file_name):
                        make_rain_image(path + file_name)

if __name__ == '__main__':
    make_prediction_image(model_name='model2', time_span=60)
    #make_prediction_image(model_name='model3')