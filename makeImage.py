# small size color map
# import color bar
import cmocean
# import pykrige
from pykrige.ok import OrdinaryKriging
from pykrige.kriging_tools import write_asc_grid
import pykrige.kriging_tools as kt
# import basemap
from mpl_toolkits.basemap import  Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Path, PathPatch
# import other main library
import csv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import random
import gc
import tracemalloc

# check memory
def format_bytes(size):
    power = 2 ** 10
    n = 0
    power_labels = ["B", "KB", "MB", "GB", "TB"]
    while size > power and n <= len(power_labels):
        size /= power
        n += 1
    return "current used memory: {:.3f} {}".format(size, power_labels[n])

def log_memory():
    snapshot = tracemalloc.take_snapshot()
    size = sum([stat.size for stat in snapshot.statistics('filename')])
    print(format_bytes(size))


## make new directory
def makeDirectory(newDirectoryPath):
    try:
        os.mkdir(newDirectoryPath)
        print("{} has successfully made!!!".format(newDirectoryPath))
    except FileExistsError:
        print("{} has already exist!!!".format(newDirectoryPath))
        pass
    return print("------------------------------------------------------------")


# make Image for Study
def imageForStudy(saveStudyImagePath, saveKriggedCSVPath, lon, lat, rain_val):
    # generate grid
    grid_lat = np.round(np.linspace(14.350, 14.760, 75), decimals=3)
    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
    # kriging
    OK = OrdinaryKriging(lon, lat, rain_val, variogram_model="exponential", verbose=True, enable_plotting=False, nlags=20)
    z1, ss1 = np.round(OK.execute("grid", grid_lon, grid_lat), decimals=3)
    xintrip, yintrip = np.meshgrid(grid_lon, grid_lat)
    fig, ax = plt.subplots(figsize=(322/96, 545/96), linewidth=0, edgecolor='w')
    #fig = plt.figure(linewidth=0, edgecolor='w')
    ax.set_xticks([])
    ax.set_yticks([])
    m = Basemap(120.90, 14.350, 121.150, 14.760, projection="merc", resolution="h", area_thresh=1000., ax=ax)
    x,y = m(xintrip, yintrip)
    ln,lt=m(lon, lat)
    ncols = 50
    # plot the data on the map
    cs = ax.contourf(x, y, z1, np.linspace(0, 100, ncols), extend="both", cmap=cmocean.cm.solar)
    fig.savefig(saveStudyImagePath)
    
    # save krigged data to csv file
    df = pd.DataFrame(z1, columns=grid_lon, index=grid_lat)
    df.to_csv(saveKriggedCSVPath)
    print("Krigged data {} saved".format(saveKriggedCSVPath))

    # release memory
    fig.clf()
    ax.cla()
    plt.close()
    del z1, ss1, cs, df, fig
    gc.collect()

    return print("Image For Study: {} saved!!!".format(saveStudyImagePath))

# image for show
def imageForShow(saveShowImagePath, lon, lat, rain_val):
    # generate grid
    grid_lat = np.round(np.linspace(14.350, 14.760, 75), decimals=3)
    grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
    # kriging
    OK = OrdinaryKriging(lon, lat, rain_val, variogram_model="exponential", verbose=True, enable_plotting=False, nlags=20)
    z1, ss1 = np.round(OK.execute("grid", grid_lon, grid_lat), decimals=3)
    xintrip, yintrip = np.meshgrid(grid_lon, grid_lat)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xticks([])
    ax.set_yticks([])
    m = Basemap(120.90, 14.350, 121.150, 14.760, projection="merc", resolution="h", area_thresh=1000., ax=ax)
    x,y = m(xintrip, yintrip)
    ln,lt=m(lon, lat)
    ncols = 50
    # plot the data on the map
    cs = ax.contourf(x, y, z1, np.linspace(0, 100, ncols), extend="both", cmap=cmocean.cm.solar)
    #save Image For Show
    fig.patch.set_facecolor('white')
    m.drawcoastlines()
    # plot the colorbar on the map
    cbar = m.colorbar(cs, location="right", pad="7%")
    # map the observation points
    x_ob, y_ob = m(lon, lat)
    m.scatter(x_ob, y_ob, marker="D", color="m")
    # draw parallels
    parallels = np.arange(14.350, 14.760, 0.050)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize="large", linewidth="0.0")
    # draw merdians
    meridians = np.arange(120.90, 121.150, 0.050)
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize="large", linewidth="0.0")
    fig.savefig(saveShowImagePath)

    # release memory
    fig.clf()
    ax.cla()
    plt.close()
    del z1, ss1, cs, fig
    gc.collect()

    return print("Image For Show: {} saved!!!".format(saveShowImagePath))

# LINE nortification
import requests
def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'HvPqtdmp53Cl6tZyKMIVkMjmBOWOWGyR6W7FG5Np31y'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)



# run program here
tracemalloc.start()
try:
    filePath = "../RafData-ForKriging/"
    rootFolderNames = os.listdir(filePath)
    for rootfolder in rootFolderNames:
        if "2019" in rootfolder or "2020" in rootfolder:
            folderPath = filePath + rootfolder + "/" # rootfolder like "2020-04-05"
            # make directory for saving image 
            newDirectoryPathForStudy = "../SolarImage-ForML/" + rootfolder + "/"
            newDirectoryPathForShow = "../SolarImage-ForShow/" + rootfolder + "/"
            newDirectoryPathForKrigged = "../SolarImage-Krigged/" + rootfolder + "/"
            makeDirectory(newDirectoryPathForStudy)
            makeDirectory(newDirectoryPathForShow)
            makeDirectory(newDirectoryPathForKrigged)
            
            dataFileNames = os.listdir(folderPath)
            if len(dataFileNames) > 0:
                for dataFile in dataFileNames:
                    # read csv and get data
                    rain_val = []
                    lat = []
                    lon = []
                    dataFilePath = folderPath + dataFile
                    with open(dataFilePath) as f:
                        reader = csv.reader(f)
                        l = [row for row in reader]
                        f.close()
                    for item in l[1:]: 
                        if float(item[2]) == 0:
                            rain_val.append(random.uniform(0.1, 0.2))
                        else:
                            rain_val.append(float(item[2]))
                        lon.append(float(item[3]))
                        lat.append(float(item[4]))
                    # If all rain_val value are the same (like all value are 0), a ValueError causes. 
                    # To avoid this, we need to create dummy data....
                    if sum(rain_val) == 0:
                        rain_val = [random.uniform(0.1, 0.2) for i in range(len(rain_val))]

                    # save Image For study
                    saveKriggedCSVPath = newDirectoryPathForKrigged + dataFile.replace(".csv", "-Krigged.csv")
                    saveStudyImagePath = newDirectoryPathForStudy + dataFile.replace(".csv", "-ForStudy.png")
                    print("-------------------------------------MAKING IMAGE FOR STUDY----------------------------------------------")
                    imageForStudy(saveStudyImagePath, saveKriggedCSVPath, lon, lat, np.round(rain_val, decimals=3))

                    # save Image For Show
                    saveShowImagePath = newDirectoryPathForShow +dataFile.replace(".csv", "-ForShow.png")
                    print("-------------------------------------MAKING IMAGE FOR SHOW----------------------------------------------")
                    imageForShow(saveShowImagePath, lon, lat, np.round(rain_val, decimals=3))

                    # console log used memory
                    print("-------------------------------------SHOW USED MEMORY-----------------------------------------------")
                    del rain_val, lat, lon, saveKriggedCSVPath, saveStudyImagePath, saveShowImagePath
                    log_memory()
            else:
                continue
    send_line_notify("Process has successfully ended!!!")
except:
    import traceback
    send_line_notify("Process has Stopped with some error!!!")
    send_line_notify(traceback.format_exc())
    print(traceback.format_exc())