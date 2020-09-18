# -*- coding: utf-8 -*-
"""
Geometric example
=================
A small example script showing the usage of the 'geographic' coordinates type
for ordinary kriging on a sphere.
"""

from pykrige.ok import OrdinaryKriging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob
from pykrige.kriging_tools import write_asc_grid
import pykrige.kriging_tools as kt
import os
os.environ['PROJ_LIB'] = r'C:\Users\tidem\Anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'


from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Path, PathPatch
import csv

with open('./2020-09-01Tempdata/temp.csv', 'r') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    f.close()

rain = []
lon = []
lat = []
for item in l[1:]:
    rain.append(float(item[1]))
    lon.append(float(item[3]))
    lat.append(float(item[2]))

# Generate a regular grid with 60° longitude and 30° latitude steps:
grid_lat = np.linspace(14.310, 14.850, 100)
grid_lon = np.linspace(120.750, 121.250, 100)

OK = OrdinaryKriging(lon, lat, rain, variogram_model='gaussian', verbose=True, enable_plotting=False,nlags=20)
z1, ss1 = OK.execute('grid', grid_lon, grid_lat)

xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)
fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(120.750, 14.310, 121.250, 14.850, projection='merc', resolution='h',area_thresh=1000.,ax=ax)
m.drawcoastlines() #draw coastlines on the map
x,y=m(xintrp, yintrp) # convert the coordinates into the map scales
ln,lt=m(lon,lat)
ncols = 50
cs=ax.contourf(x, y, z1, np.linspace(min(rain)-1, max(rain)+1, ncols),extend='both',cmap='jet') #plot the data on the map.
cbar=m.colorbar(cs,location='right',pad="7%") #plot the colorbar on the map
# draw parallels.
parallels = np.arange(14.310,14.850,0.1)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=14, linewidth=0.0) #Draw the latitude labels on the map
 
# draw meridians
meridians = np.arange(120.750,121.250,0.1)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14, linewidth=0.0)