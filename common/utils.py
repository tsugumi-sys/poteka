from typing import Generator, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tracemalloc


def datetime_range(start: datetime, end: datetime, delta: timedelta) -> Generator[datetime, None, None]:
    current = start
    while current <= end:
        yield current
        current += delta


def make_dates(x):
    if len(str(x)) == 2:
        return str(x)
    else:
        return "0" + str(x)


def csv_list(year: int = 2020, month: int = 1, date: int = 1) -> List[str]:
    dts = [
        f"{dt.hour}-{dt.minute}.csv"
        for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))
    ]
    return dts


def format_bytes(size: int) -> str:
    power = 2 ** 10
    n = 0
    power_labels = ["B", "KB", "MB", "GB", "TB"]
    while size > power and n <= len(power_labels):
        size /= power
        n += 1
    return f"current used memory: {size} {power_labels[n]}"


def log_memory() -> None:
    snapshot = tracemalloc.take_snapshot()
    size = sum([stat.size for stat in snapshot.statistics("filename")])
    print(format_bytes(size))


def min_max_scaler(min_value: float, max_value: float, arr: np.ndarray):
    return (arr - min_value) / (max_value - min_value)


def rescale_arr(min_value: float, max_value: float, arr: np.ndarray):
    return (max_value - min_value) * arr + min_value


# return: ndarray
def load_csv_data(path: str):
    df = pd.read_csv(path, index_col=0)
    if "rain" in path:
        # df = df + 50
        # Scale [0, 100]
        return min_max_scaler(0, 100, df.values)

    elif "temp" in path:
        # Scale [10, 45]
        return min_max_scaler(10, 45, df.values)

    elif "abs_wind" in path:
        df = np.where(df > 15, 15, df)
        return min_max_scaler(0, 15, df.values)

    elif "wind" in path:
        # Scale [-10, 10]
        return min_max_scaler(-10, 10, df.values)

    elif "humidity" in path:
        return min_max_scaler(0, 100, df.values)

    elif "pressure" in path:
        return min_max_scaler(990, 1025, df.values)


def get_param_path(param_name: str, year, month, date) -> Optional[str]:
    if "rain" in param_name:
        return f"../../../data/rain_image/{year}/{month}/{date}"
    elif "abs_wind" in param_name:
        return f"../../../data/abs_wind_image/{year}/{month}/{date}"
    elif "wind" in param_name:
        return f"../../../data/wind_image/{year}/{month}/{date}"
    elif "temperature" in param_name:
        return f"../../../data/temp_image/{year}/{month}/{date}"
    elif "humidity" in param_name:
        return f"../../../data/humidity_image/{year}/{month}/{date}"
    elif "station_pressure" in param_name:
        return f"../../../data/station_pressure_image/{year}/{month}/{date}"
    elif "seaLevel_pressure" in param_name:
        return f"../../../data/seaLevel_pressure_image/{year}/{month}/{date}"
    else:
        print(param_name, "is wrong or spell missing.")
        return


def chack_data_scale(data: np.ndarray) -> bool:
    if data.max() > 1 or data.min() < 0:
        return False
    else:
        return True


def create_time_list(year: int = 2020, month: int = 1, date: int = 1, delta: int = 10) -> List[datetime]:
    dts = [dt for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=delta))]
    return dts
