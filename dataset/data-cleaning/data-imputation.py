import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="mean")

rain_cols = ["RAF", "RA1", "RI1", "ERA", "CRA"]
cols = ["RAF", "RA1", "RI1", "ERA", "CRA", "AT1", "RH1", "PRS", "SLP", "WD1", "WDM", "WS1", "WSM", "WND", "WNS", "SOL", "WET", "WBG", "WEA"]

folder_path = "../../../data/cleaned-data"
ob_folders = os.listdir(folder_path)
for ob_folder in ob_folders:
    folders = os.listdir(folder_path + f"/{ob_folder}")
    if folders:
        for year_folder in folders:
            month_folders = os.listdir(folder_path + f"/{ob_folder}/{year_folder}")
            if month_folders:
                for month_folder in month_folders:
                    data_folders = os.listdir(folder_path + f"/{ob_folder}/{year_folder}/{month_folder}")
                    if data_folders:
                        for data_folder in data_folders:
                            data_file = folder_path + f"/{ob_folder}/{year_folder}/{month_folder}/{data_folder}/data.csv"
                            if os.path.exists(data_file):
                                print("-" * 60)
                                print(data_file)
                                df = pd.read_csv(data_file, index_col="Datetime")
                                if len(df) != 1440:
                                    print(len(df), data_file)
                                else:
                                    total_missing = df.isnull().sum().sum()
                                    imp_df = df.copy()
                                    imp_df = imp_df.fillna(method="bfill", axis=0, limit=3)
                                    imp_df = imp_df.fillna(method="ffill", axis=0, limit=3)
                                    imp_df[rain_cols] = imp_df[rain_cols].fillna(0)
                                    imp_df = pd.DataFrame(imp.fit_transform(imp_df))
                                    imp_df.columns = df.columns
                                    imp_df.index = df.index

                                    # check nan value
                                    total_missing_imputed = imp_df.isnull().sum().sum()
                                    total_cells = np.product(imp_df.shape)
                                    print("Percentage of nan cells")
                                    print("Before: ", (total_missing / total_cells) * 100)
                                    print("After: ", (total_missing_imputed / total_cells) * 100)

                                    # save
                                    save_path = "../../../data/imputed-data"
                                    f = [ob_folder, year_folder, month_folder, data_folder]
                                    for i in f:
                                        save_path = save_path + f"/{i}"
                                        print(save_path)
                                        if not os.path.exists(save_path):
                                            os.mkdir(save_path)
                                    save_path = save_path + "/data.csv"
                                    imp_df.to_csv(save_path)
