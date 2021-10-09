import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import numpy as np
from utils import csv_list


# Calculate RMSE of whole period
def calc_period_rmse(model_type: str, model_name: str, time_span: int) -> None:
    # load validation date and time
    valid_list = pd.read_csv("valid_data_list.csv")

    # Create csv list
    csv_files = csv_list(2020, 1, 1)

    root = f"../../data/prediction_image/{model_type}/{time_span}min_{model_name}/2019/"
    rmses = []
    dates = valid_list["date"].unique()
    for date in dates:

        month = date.split("-")[1]
        rmse_df = pd.read_csv(root + month + "/RMSE/" + date + ".csv")

        files = valid_list.loc[valid_list["date"] == date]
        for i in files.index:
            start, end = valid_list.loc[i, "start"], valid_list.loc[i, "end"]
            idx_start, idx_end = csv_files.index(start), csv_files.index(end)
            idx_end = idx_end + 1
            rmses.append(rmse_df[idx_start:idx_end]["RMSE"].mean())

    print("=" * 60)
    print(model_type, model_name, time_span, "RMSE", sum(rmses) / len(rmses))


# Calculate RMSE of each time and day
def calc_rmse(model_type: str = "oneByone_model", model_name: str = "model1", time_span: int = 60) -> None:
    print("-" * 60)
    print(model_type, model_name, f"{time_span}min")
    print("-" * 60)
    csv_files = csv_list(2020, 1, 1)
    monthes = ["10", "11"]
    prediction_root = f"../../data/prediction_image/{model_type}/{time_span}min_{model_name}/2019/"
    label_root = "../../data/rain_image/2019/"

    for month in monthes:
        if not os.path.exists(prediction_root + month + "/RMSE/"):
            os.mkdir(prediction_root + month + "/RMSE/")

        for date in os.listdir(label_root + month):
            rmse_df = pd.DataFrame()
            print("Calculating RMSE ...", date)
            for csv_file in csv_files[6:]:
                label_data = pd.read_csv(label_root + month + f"/{date}/{csv_file}", index_col=0)
                pred_data = pd.read_csv(prediction_root + month + f"/{date}/{csv_file}", index_col=0)
                rmse = ((label_data.values - pred_data.values) ** 2).mean() ** 0.5
                rmse_df.loc[csv_file, "RMSE"] = rmse

            rmse_df.to_csv(prediction_root + month + f"/RMSE/{date}.csv")


def create_error_graph(model_type: str, model_name: str, time_span: int, date: str, start_time: str) -> None:
    print(model_type, model_name, time_span, date, start_time, "Creating ...")
    year, month = date.split("-")[0], date.split("-")[1]
    root_dir = f"../../data/prediction_image/{model_type}/{time_span}min_{model_name}/{year}/{month}/{date}/Valid_Data/"

    diff_pred_label_df = pd.DataFrame()
    diff_pred_krigLabel_df = pd.DataFrame()

    csv_files = csv_list(2020, 1, 1)
    start_idx = csv_files.index(start_time + ".csv")
    next_idx = start_idx + 6

    for csv in csv_files[start_idx:next_idx]:
        df = pd.read_csv(root_dir + csv, index_col=0)
        df["diff_pred_Label"] = abs(df["Pred"] - df["Label"])
        df["diff_pred_krigLabel"] = abs(df["Pred"] - df["Krig_Label"])

        time = csv.split(".")[0]

        for i in df.index:
            diff_pred_label_df.loc[time, i] = df.loc[i, "diff_pred_krigLabel"]
            diff_pred_krigLabel_df.loc[time, i] = df.loc[i, "diff_pred_krigLabel"]

    new_cols = [f"point{i}" for i in range(1, len(diff_pred_label_df.columns) + 1)]
    diff_pred_label_df.columns = new_cols
    diff_pred_krigLabel_df.columns = new_cols

    diff_pred_label_df.index.name = "time"
    diff_pred_label_df.index = diff_pred_label_df.index.map(lambda x: x + "UTC")

    for i in diff_pred_label_df.index:
        diff_pred_label_df.loc[i, "Max"] = diff_pred_label_df.loc[i, :].max()
        diff_pred_label_df.loc[i, "Mean"] = diff_pred_label_df.loc[i, :].mean()
        diff_pred_label_df.loc[i, "Min"] = diff_pred_label_df.loc[i, :].min()
        diff_pred_label_df.loc[i, "Std"] = diff_pred_label_df.loc[i, :].std()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=diff_pred_label_df[["Mean"]], palette="Blues_r", lw=1.5, ax=ax, dashes=False, markers=True)
    x = diff_pred_label_df.index
    y_lower = diff_pred_label_df["Mean"] - diff_pred_label_df["Std"]
    y_upper = diff_pred_label_df["Mean"] + diff_pred_label_df["Std"]
    ax.fill_between(x, y_lower, y_upper, alpha=0.2)
    ax.set_title("The error between the prediction values and the label values")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error (mm/h)")
    plt.savefig(root_dir + f"{date}.png")
    plt.close()


def make_rain_image(path: str) -> None:
    original_df = pd.read_csv("../../p-poteka-config/observation_point.csv", index_col="Name")
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

    clevs = [0, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]
    cmap_data = [
        (1.0, 1.0, 1.0),
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
    cmap = mcolors.ListedColormap(cmap_data, "precipitation")
    norm = mcolors.BoundaryNorm(clevs, cmap.N)

    cs = ax.contourf(xi, yi, data, clevs, cmap=cmap, norm=norm)
    cbar = plt.colorbar(cs, orientation="vertical")
    cbar.set_label("millimeter")
    ax.scatter(original_df["LON"], original_df["LAT"], marker="D", color="dimgrey")

    save_path = path.replace(".csv", ".png")
    plt.savefig(save_path)
    plt.close()
    print(save_path)
    print("Sucessfully Saved")


def make_prediction_image(model_type: str = "oneByone_model", model_name: str = "model1", time_span: str = 60):
    root_path = f"../../data/prediction_image/{model_type}/{time_span}min_{model_name}"
    for year in os.listdir(root_path):
        for month in os.listdir(root_path + f"/{year}"):
            for date in os.listdir(root_path + f"/{year}/{month}"):
                if date != "RMSE":
                    path = root_path + f"/{year}/{month}/{date}/"
                    csv_files = [file_name for file_name in os.listdir(path) if ".csv" in file_name]
                    print(csv_files)
                    for file_name in csv_files:
                        if os.path.exists(path + file_name):
                            make_rain_image(path + file_name)


def create_valid_each_point(model_type: str, model_name: str, time_span: int) -> None:
    root_dir = f"../../data/prediction_image/{model_type}/{time_span}min_{model_name}/"
    years = ["2019"]
    print("*" * 100)
    print(f"{model_type}/{time_span}min_{model_name}/")
    print("*" * 100)
    for year in years:
        for month in os.listdir(root_dir + year):
            for date in [f for f in os.listdir(root_dir + year + f"/{month}") if not f == "RMSE"]:
                print(date, "Create and Saving ...")
                for csv in [f for f in os.listdir(root_dir + year + f"/{month}/{date}") if ".csv" in f]:
                    point_data_path = "../../p-poteka-config/observation_point.csv"
                    oneday_data_path = f"../../data/one_day_data/{year}/{month}/{date}/{csv}"
                    label_data_path = f"../../data/rain_image/{year}/{month}/{date}/{csv}"
                    pred_data_path = root_dir + f"{year}/{month}/{date}/{csv}"

                    point_data_df = pd.read_csv(point_data_path, index_col=0)
                    oneday_data_df = pd.read_csv(oneday_data_path, index_col=0)
                    label_data_df = pd.read_csv(label_data_path, index_col=0)
                    pred_data_df = pd.read_csv(pred_data_path, index_col=0)

                    grid_lons, grid_lats = label_data_df.columns.values.astype(float).tolist(), label_data_df.index.values.astype(float).tolist()

                    for i in point_data_df.index:
                        lon, lat = point_data_df.loc[i, "LON"], point_data_df.loc[i, "LAT"]
                        prev_grid_lon, prev_grid_lat = grid_lons[0], grid_lats[0]
                        idx_lon, ldx_lat = 0, 0

                        for grid_lon in grid_lons[1:]:
                            if grid_lon > lon and prev_grid_lon < lon:
                                idx_lon = grid_lons.index(grid_lon)
                                break
                            prev_grid_lon = grid_lon

                        for grid_lat in grid_lats[1:]:
                            if grid_lat > lat and prev_grid_lat > lat:
                                idx_lat = grid_lats.index(grid_lat)
                                break
                            prev_grid_lat = grid_lat

                        point_data_df.loc[i, "Pred"] = pred_data_df.iloc[idx_lon, ldx_lat]
                        point_data_df.loc[i, "Krig_Label"] = label_data_df.iloc[idx_lon, idx_lat]
                        if i in oneday_data_df.index:
                            point_data_df.loc[i, "Label"] = oneday_data_df.loc[i, "hour-rain"]
                        else:
                            point_data_df.loc[i, "Label"] = np.nan

                    # save data
                    save_path = root_dir + f"{year}/{month}/{date}/Valid_Data/"
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    point_data_df.to_csv(save_path + csv)
