from typing import Union


def is_ymd_valid(
    year: Union[str, int], month: Union[str, int], date: Union[str, int], data_file_path: str,
):
    if type(year) is int or type(month) is int or type(date) is int:
        year, month, date = str(year), str(month), str(date)
    return year in data_file_path and month in data_file_path and date in data_file_path
