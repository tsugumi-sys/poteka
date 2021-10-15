import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RBFInterpolator
from matplotlib import cm
import argparse
import sys
from typing import Union
from logging import getLogger, INFO, basicConfig, StreamHandler
import multiprocessing
from joblib import Parallel, delayed
from utils import gen_data_config

sys.path.append(".")
from common.send_info import send_line  # noqa: E402
from common.validations import is_ymd_valid  # noqa: E402


logger = getLogger(__name__)
logger.setLevel(INFO)
basicConfig(
    level=INFO,
    filename="./dataset/data-making/log/create_humidity_data.log",
    filemode="w",
    format="%(asctime)s %(levelname)s %(name)s :%(message)s",
)
logger.addHandler(StreamHandler(sys.stdout))


def make_img(
    data_file_path: str,  # something like ../data/one_day_data/{year}/{month}/{date}/{hour}-{minute}.csv
    csv_file_name: str,  # {hour}-{minute}.csv
    save_dir_path: str,  # something like ../data/station_pressure_image
    year: Union[str, int],
    month: Union[str, int],
    date: Union[str, int],
) -> None:
    basicConfig(
        level=INFO,
        filename="./dataset/data-making/log/create_humidity_data.log",
        filemode="a",
        format="%(asctime)s %(levelname)s %(name)s :%(message)s",
    )

    img_title = "Relative Humidity"
    is_data_file_exists = os.path.exists(data_file_path)
    is_save_dir_exists = os.path.exists(save_dir_path)

    if is_data_file_exists and is_save_dir_exists and is_ymd_valid(year, month, date, data_file_path):
        try:
            df = pd.read_csv(data_file_path, index_col=0)
            rbfi = RBFInterpolator(
                y=df[["LON", "LAT"]],
                d=df["RH1"],
                kernel="linear",
                epsilon=10,
            )
            grid_lon = np.round(np.linspace(120.90, 121.150, 50), decimals=3)
            grid_lat = np.round(np.linspace(14.350, 14.760, 50), decimals=3)
            # xi, yi = np.meshgrid(grid_lon, grid_lat)
            xgrid = np.around(
                np.mgrid[120.90:121.150:50j, 14.350:14.760:50j],
                decimals=3,
            )
            xfloat = xgrid.reshape(2, -1).T

            z1 = rbfi(xfloat)
            z1 = z1.reshape(50, 50)
            humid_data = np.where(z1 > 0, z1, 0)
            humid_data = np.where(humid_data > 100, 100, humid_data)

            plt.figure(figsize=(7, 8), dpi=80)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([120.90, 121.150, 14.350, 14.760])
            ax.add_feature(cfeature.COASTLINE)
            gl = ax.gridlines(draw_labels=True, alpha=0)
            gl.right_labels = False
            gl.top_labels = False

            clevs = [i for i in range(0, 101, 5)]

            cmap = cm.Blues
            norm = mcolors.BoundaryNorm(clevs, cmap.N)

            cs = ax.contourf(*xgrid, humid_data, clevs, cmap=cmap, norm=norm)
            cbar = plt.colorbar(cs, orientation="vertical")
            cbar.set_label("%")
            ax.scatter(
                df["LON"],
                df["LAT"],
                marker="D",
                color="dimgrey",
            )
            for i, val in enumerate(df["RH1"]):
                ax.annotate(val, (df["LON"][i], df["LAT"][i]))
            ax.set_title(img_title)

            # Save Image and CSV
            save_path = save_dir_path
            folders = [year, month, date]
            for folder in folders:
                if not os.path.exists(save_path + f"/{folder}"):
                    os.mkdir(save_path + f"/{folder}")
                save_path += f"/{folder}"
            save_csv_path = save_path + f"/{csv_file_name}"
            save_path += "/{}".format(csv_file_name.replace(".csv", ".png"))
            plt.savefig(save_path)

            save_df = pd.DataFrame(humid_data)
            save_df = save_df[save_df.columns[::-1]].T
            save_df.columns = grid_lon
            save_df.index = grid_lat[::-1]
            save_df.to_csv(save_csv_path)

            plt.close()
        except:
            logger.exception(f"Creating data of {data_file_path} has failed with some erors")
    else:
        if not is_data_file_exists:
            logger.error("data_file_path: %s does not exist.", data_file_path)
        elif not is_save_dir_exists:
            logger.error("save_dir_path: %s does not exist.", save_dir_path)
        else:
            logger.error("Year: %s, Month: %s, Date: %s does not match with %s", year, month, date, data_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process humidity data.")
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="../../../data",
        help="The root path of the data directory.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="The number of cpus to use.",
    )

    args = parser.parse_args()

    save_dir_name = "humidity_image"
    confs = gen_data_config(data_root_path=args.data_root_path, save_dir_name=save_dir_name)
    n_jobs = args.n_jobs

    max_cores = multiprocessing.cpu_count()
    if n_jobs > max_cores:
        n_jobs = max_cores

    Parallel(n_jobs=n_jobs)(
        delayed(make_img)(
            data_file_path=conf["data_file_path"],
            csv_file_name=conf["csv_file_name"],
            save_dir_path=conf["save_dir_path"],
            year=conf["year"],
            month=conf["month"],
            date=conf["date"],
        )
        for conf in confs[:10]
    )

    send_line("Creating humidity data has finished.")
