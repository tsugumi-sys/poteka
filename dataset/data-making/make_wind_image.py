# from metpy.units import units
# from metpy.calc import wind_components
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import RBFInterpolator
from matplotlib import cm
import sys
import argparse
from typing import Union
from logging import getLogger, INFO, basicConfig, StreamHandler
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import gen_data_config, tqdm_joblib

sys.path.append(".")
from common.send_info import send_line  # noqa: E402
from common.validations import is_ymd_valid  # noqa: E402


logger = getLogger(__name__)
logger.setLevel(INFO)
basicConfig(
    level=INFO, filename="./dataset/data-making/log/create_wind_data.log", filemode="w", format="%(asctime)s %(levelname)s %(name)s :%(message)s",
)
logger.addHandler(StreamHandler(sys.stdout))


def calc_u_v(df, ob_point):
    wind_dir = float(df["WD1"])
    wind_speed = float(df["WS1"])

    rads = np.radians(float(wind_dir))
    wind_u, wind_v = -1 * wind_speed * np.cos(rads), -1 * wind_speed * np.sin(rads)
    # wind_u_v = wind_components(wind_speed * units("m/s"), wind_dir * units.deg)

    return [
        ob_point,
        round(wind_u, 5),
        round(wind_v, 5),
    ]  # (index, u wind, v wind) u: X (East-West) v: Y(North-South)


def make_abs_img(
    data_file_path: str,  # something like ../data/one_day_data/{year}/{month}/{date}/{hour}-{minute}.csv
    csv_file_name: str,  # {hour}-{minute}.csv
    save_dir_path: str,  # something like ../data/station_pressure_image
    year: Union[str, int],
    month: Union[str, int],
    date: Union[str, int],
) -> None:
    basicConfig(
        level=INFO, filename="./dataset/data-making/log/create_wind_data.log", filemode="a", format="%(asctime)s %(levelname)s %(name)s :%(message)s",
    )

    img_title = "wind speed (meter/second)"
    is_data_file_exists = os.path.exists(data_file_path)
    is_save_dir_exists = os.path.exists(save_dir_path)

    if is_data_file_exists and is_save_dir_exists and is_ymd_valid(year, month, date, data_file_path):
        try:
            df = pd.read_csv(data_file_path, index_col=0)

            # Interpolate Data
            grid_size = 50
            wind_rbfi = RBFInterpolator(y=df[["LON", "LAT"]], d=df["WS1"], kernel="linear", epsilon=10,)

            grid_lon = np.round(np.linspace(120.90, 121.150, grid_size), decimals=3,)
            grid_lat = np.round(np.linspace(14.350, 14.760, grid_size), decimals=3,)
            # xi, yi = np.meshgrid(grid_lon, grid_lat)
            xgrid = np.around(np.mgrid[120.90:121.150:50j, 14.350:14.760:50j], decimals=3,)
            xfloat = xgrid.reshape(2, -1).T

            z1 = wind_rbfi(xfloat)
            z1 = z1.reshape(50, 50)
            abs_wind = np.where(z1 > 30, 30, z1)
            abs_wind = np.where(abs_wind < 0, 0, abs_wind)

            # Save Fig
            plt.figure(figsize=(7, 8), dpi=80)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([120.90, 121.150, 14.350, 14.760])
            ax.add_feature(cfeature.COASTLINE)
            gl = ax.gridlines(draw_labels=True, alpha=0)
            gl.right_labels = False
            gl.top_labels = False

            clevs = list(range(0, 31, 2))
            cmap = cm.viridis
            norm = mcolors.BoundaryNorm(clevs, cmap.N)

            cs = ax.contourf(*xgrid, abs_wind, clevs, cmap=cmap, norm=norm)
            cbar = plt.colorbar(cs, orientation="vertical")
            cbar.set_label(img_title)
            ax.scatter(
                df["LON"], df["LAT"], marker="D", color="dimgrey",
            )
            for i, val in enumerate(df["WS1"]):
                ax.annotate(val, (df["LON"][i], df["LAT"][i]))

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

            save_df = pd.DataFrame(abs_wind)
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


def make_uv_img(
    data_file_path: str,  # something like ../data/one_day_data/{year}/{month}/{date}/{hour}-{minute}.csv
    csv_file_name: str,  # {hour}-{minute}.csv
    save_dir_path: str,  # something like ../data/station_pressure_image
    year: Union[str, int],
    month: Union[str, int],
    date: Union[str, int],
) -> None:
    basicConfig(
        level=INFO, filename="./dataset/data-making/log/create_wind_data.log", filemode="a", format="%(asctime)s %(levelname)s %(name)s :%(message)s",
    )

    img_title = "wind speed (m/second)"
    is_data_file_exists = os.path.exists(data_file_path)
    is_save_dir_exists = os.path.exists(save_dir_path)

    if is_data_file_exists and is_save_dir_exists and is_ymd_valid(year, month, date, data_file_path):
        try:
            df = pd.read_csv(data_file_path, index_col=0)
            wind_df = pd.DataFrame([calc_u_v(df.loc[i, :], i) for i in df.index], columns=["OB-POINT", "U-WIND", "V-WIND"],)
            wind_df = wind_df.set_index("OB-POINT")
            wind_df["LON"] = df["LON"]
            wind_df["LAT"] = df["LAT"]
            grid_size = 50
            v_wind_rbfi = RBFInterpolator(y=wind_df[["LON", "LAT"]], d=wind_df["V-WIND"], kernel="linear", epsilon=10,)
            u_wind_rbfi = RBFInterpolator(y=wind_df[["LON", "LAT"]], d=wind_df["U-WIND"], kernel="linear", epsilon=10,)

            grid_lon = np.round(np.linspace(120.90, 121.150, grid_size), decimals=3,)
            grid_lat = np.round(np.linspace(14.350, 14.760, grid_size), decimals=3,)
            # xi, yi = np.meshgrid(grid_lon, grid_lat)
            xgrid = np.around(np.mgrid[120.90:121.150:50j, 14.350:14.760:50j], decimals=3,)
            xfloat = xgrid.reshape(2, -1).T

            z_v_wind = v_wind_rbfi(xfloat)
            z_u_wind = u_wind_rbfi(xfloat)
            z_v_wind = z_v_wind.reshape(50, 50)
            z_u_wind = z_u_wind.reshape(50, 50)

            v_wind = np.where(z_v_wind > 10, 10, z_v_wind)
            v_wind = np.where(v_wind < -10, -10, v_wind)
            u_wind = np.where(z_u_wind > 10, 10, z_u_wind)
            u_wind = np.where(u_wind < -10, -10, u_wind)

            # Calculate divergence
            # v_wind_grad = np.array(np.gradient(v_wind)[1])
            # u_wind_grad = np.array(np.gradient(u_wind)[0])
            # wind_div = np.empty([grid_size, grid_size])
            # for i in range(grid_size):
            #     for j in range(grid_size):
            #         x_left = i - 1 if i - 1 > 0 else 0
            #         x_right = i + 1 if i + 1 < grid_size - 1 else grid_size - 1
            #         y_above = j + 1 if j + 1 < grid_size - 1 else grid_size - 1
            #         y_bottom = j - 1 if j - 1 > 0 else 0
            #         val = 0
            #         for x in range(x_left, x_right + 1):
            #             for y in range(y_bottom, y_above + 1):
            #                 val += v_wind_grad[x, y] + u_wind_grad[x, y]
            #         wind_div[i, j] = val
            # wind_div = np.where(wind_div > 10, 10, wind_div)
            # wind_div = np.where(wind_div < -10, -10, wind_div)
            # print(wind_div.max(), wind_div.min())

            # Save Image and CSV
            save_path = save_dir_path
            folders = [year, month, date]
            for folder in folders:
                if not os.path.exists(save_path + f"/{folder}"):
                    os.mkdir(save_path + f"/{folder}")
                save_path += f"/{folder}"
            save_csv_path = save_path + f"/{csv_file_name}"
            save_path += "/{}".format(csv_file_name.replace(".csv", ".png"))
            save_u_wind_fig_path = save_path.replace(".png", "U.png")
            save_v_wind_fig_path = save_path.replace(".png", "V.png")

            dic = {
                "U-Wind": {"save_path": save_u_wind_fig_path, "data": u_wind,},
                "V-Wind": {"save_path": save_v_wind_fig_path, "data": v_wind,},
            }

            for key in list(dic.keys()):
                plt.figure(figsize=(7, 8), dpi=80)
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

                cs = ax.contourf(*xgrid, dic[key]["data"], clevs, cmap=cmap, norm=norm,)
                cbar = plt.colorbar(cs, orientation="vertical")
                cbar.set_label(img_title)
                ax.set_title(key)
                # plt.quiver(xi, yi, u_wind, v_wind)
                ax.scatter(
                    df["LON"], df["LAT"], marker="D", color="dimgrey",
                )
                for i, val in enumerate(wind_df[key.upper()]):
                    ax.annotate(val, (df["LON"][i], df["LAT"][i]))
                plt.savefig(dic[key]["save_path"])
                plt.close()

            uwind_df = pd.DataFrame(u_wind)
            vwind_df = pd.DataFrame(v_wind)
            uwind_df = uwind_df[uwind_df.columns[::-1]].T
            vwind_df = vwind_df[vwind_df.columns[::-1]].T
            uwind_df.columns = grid_lon
            vwind_df.columns = grid_lon
            uwind_df.index = grid_lat[::-1]
            vwind_df.index = grid_lat[::-1]
            uwind_df.to_csv(save_csv_path.replace(".csv", "U.csv"))
            vwind_df.to_csv(save_csv_path.replace(".csv", "V.csv"))
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
        "--data_root_path", type=str, default="../../../data", help="The root path of the data directory.",
    )

    parser.add_argument(
        "--n_jobs", type=int, default=1, help="The number of cpus to use.",
    )

    parser.add_argument(
        "--target", type=str, default="abs", help="Taget name (abs or uv).",
    )

    args = parser.parse_args()

    target = args.target

    if target not in ["abs", "uv"]:
        logger.error('--target shoud be "abs" or "uv"')
    else:
        save_dir_name = "abs_wind_image" if target == "abs" else "wind_image"
        confs = gen_data_config(data_root_path=args.data_root_path, save_dir_name=save_dir_name)
        n_jobs = args.n_jobs

        max_cores = multiprocessing.cpu_count()
        if n_jobs > max_cores:
            n_jobs = max_cores

        if target == "abs":
            with tqdm_joblib(tqdm(desc="Create abs wind data", total=len(confs))):
                Parallel(n_jobs=n_jobs)(
                    delayed(make_abs_img)(
                        data_file_path=conf["data_file_path"],
                        csv_file_name=conf["csv_file_name"],
                        save_dir_path=conf["save_dir_path"],
                        year=conf["year"],
                        month=conf["month"],
                        date=conf["date"],
                    )
                    for conf in confs
                )
        else:
            with tqdm_joblib(tqdm(desc="Create uv wind data.", total=len(confs))):
                Parallel(n_jobs=n_jobs)(
                    delayed(make_uv_img)(
                        data_file_path=conf["data_file_path"],
                        csv_file_name=conf["csv_file_name"],
                        save_dir_path=conf["save_dir_path"],
                        year=conf["year"],
                        month=conf["month"],
                        date=conf["date"],
                    )
                    for conf in confs
                )

        send_line("Creating wind data has finished.")
