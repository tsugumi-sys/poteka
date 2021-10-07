import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RBFInterpolator
from matplotlib import cm
import requests
import tracemalloc
import traceback
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path("../../.env")
load_dotenv(dotenv_path=dotenv_path)


def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = os.getenv("LINE_TOKEN")
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"message: {notification_message}"}
    requests.post(line_notify_api, headers=headers, data=data)


def make_temp_image():

    tracemalloc.start()
    failed_path = []
    try:
        root_folder = "../../../data/one_day_data"

        for year in os.listdir(root_folder):
            for month in os.listdir(root_folder + f"/{year}"):
                for date in os.listdir(root_folder + f"/{year}/{month}"):
                    if len(os.listdir(root_folder + f"/{year}/{month}/{date}")) > 0:
                        data_files = os.listdir(root_folder + f"/{year}/{month}/{date}")
                        for data_file in data_files:
                            path = root_folder + f"/{year}/{month}/{date}/{data_file}"
                            if os.path.exists(path):
                                print("-" * 80)
                                print("Temperature")
                                print("PATH: ", path)
                                try:
                                    df = pd.read_csv(path, index_col=0)
                                    # bins = gs.standard_bins((df['LAT'], df['LON']), max_dist=np.deg2rad(0.5), latlon=True)
                                    # bin_c, vario = gs.vario_estimate((df['LAT'], df['LON']), df['AT1'], bins, latlon=True)
                                    # model = gs.Cubic(latlon=True, rescale=gs.EARTH_RADIUS, var=1, len_scale=1.0)
                                    # model.fit_variogram(bin_c, vario, nugget=False)
                                    # grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
                                    # grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
                                    # #z1, ss1 = np.round(OK.execute("grid", grid_lon, grid_lat), decimals=3)
                                    # OK_gs = gs.krige.Ordinary(model, [df['LAT'], df['LON']], df['AT1'], exact=True)
                                    # z1 = OK_gs.structured([grid_lat, grid_lon])
                                    # z1 = z1[0]
                                    # xintrip, yintrip = np.meshgrid(grid_lon, grid_lat)
                                    rbfi = RBFInterpolator(
                                        y=df[["LON", "LAT"]],
                                        d=df["AT1"],
                                        kernel="linear",
                                        epsilon=10,
                                    )
                                    grid_lon = np.round(
                                        np.linspace(120.90, 121.150, 50), decimals=3
                                    )
                                    grid_lat = np.round(
                                        np.linspace(14.350, 14.760, 50), decimals=3
                                    )
                                    # xi, yi = np.meshgrid(grid_lon, grid_lat)
                                    xgrid = np.around(
                                        np.mgrid[120.90:121.150:50j, 14.350:14.760:50j],
                                        decimals=3,
                                    )
                                    xfloat = xgrid.reshape(2, -1).T

                                    z1 = rbfi(xfloat)
                                    z1 = z1.reshape(50, 50)
                                    temp_data = np.where(z1 > 10, z1, 10)
                                    temp_data = np.where(z1 > 45, 45, temp_data)

                                    fig = plt.figure(figsize=(8, 8))
                                    ax = plt.axes(projection=ccrs.PlateCarree())
                                    ax.set_extent([120.90, 121.150, 14.350, 14.760])
                                    ax.add_feature(cfeature.COASTLINE)
                                    gl = ax.gridlines(draw_labels=True, alpha=0)
                                    gl.right_labels = False
                                    gl.top_labels = False

                                    clevs = [i for i in range(10, 46)]

                                    cmap = cm.rainbow
                                    norm = mcolors.BoundaryNorm(clevs, cmap.N)

                                    cs = ax.contourf(
                                        *xgrid, temp_data, clevs, cmap=cmap, norm=norm
                                    )
                                    cbar = plt.colorbar(cs, orientation="vertical")
                                    cbar.set_label("°C")
                                    ax.scatter(
                                        df["LON"],
                                        df["LAT"],
                                        marker="D",
                                        color="dimgrey",
                                    )
                                    for i, val in enumerate(df["AT1"]):
                                        ax.annotate(val, (df["LON"][i], df["LAT"][i]))
                                    ax.set_title("Temperature")

                                    # Save Image
                                    save_path = "../../../data/temp_image"
                                    folders = [year, month, date]
                                    for folder in folders:
                                        if not os.path.exists(save_path + f"/{folder}"):
                                            os.mkdir(save_path + f"/{folder}")
                                        save_path += f"/{folder}"
                                    save_csv_path = save_path + f"/{data_file}"
                                    save_path += "/{}".format(
                                        data_file.replace(".csv", ".png")
                                    )
                                    plt.savefig(save_path)
                                    plt.close()

                                    save_df = pd.DataFrame(temp_data)
                                    save_df = save_df[save_df.columns[::-1]].T
                                    save_df.columns = grid_lon
                                    save_df.index = grid_lat[::-1]
                                    save_df.to_csv(save_csv_path)
                                    print("Sucessfully Saved")
                                except:
                                    print("!" * 10, " Failed ", "!" * 10)
                                    failed_path.append(path)
                                    print(traceback.format_exc())
                                    continue
        failed = pd.DataFrame({"path": failed_path})
        failed.to_csv("failed.csv")
        send_line_notify("Succeccfuly Completed!!!")
    except:
        send_line_notify("Process has Stopped with some error!!!")
        send_line_notify(traceback.format_exc())
        print(traceback.format_exc())


if __name__ == "__main__":
    make_temp_image()
