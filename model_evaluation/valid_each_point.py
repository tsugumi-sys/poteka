import pandas as pd
import numpy as np
import os


def create_valid_each_point(model_type, model_name, time_span):
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


if __name__ == "__main__":
    models = {
        "oneByone_model": {
            "model_names": ["rth_optuna", "ruvth_optuna", "ruvthpp_optuna", "rwthpp_optuna"],
            "time_span": 60,
        }
    }

    for model_type in models.keys():
        time_span = models["oneByone_model"]["time_span"]
        for model_name in models["oneByone_model"]["model_names"]:
            create_valid_each_point(model_type, model_name, time_span)
