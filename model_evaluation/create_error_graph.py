import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta


def csv_list(year, month, date):
    dts = [
        f"{dt.hour}-{dt.minute}.csv"
        for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))
    ]
    return dts


def create_error_graph(model_type, model_name, time_span, date, start_time):
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


if __name__ == "__main__":
    models = {
        "oneByone_model": {
            "model_names": ["rth_optuna", "ruvth_optuna", "ruvthpp_optuna", "rwthpp_optuna"],
            "time_span": 60,
        }
    }

    cases = [
        {"date": "2019-10-11", "start_time": "13-0"},
        {"date": "2019-10-12", "start_time": "9-0"},
    ]

    for model_type in models.keys():
        time_span = models["oneByone_model"]["time_span"]
        for model_name in models["oneByone_model"]["model_names"]:
            for case in cases:
                date = case["date"]
                start_time = case["start_time"]
                create_error_graph(model_type, model_name, time_span, date, start_time)
