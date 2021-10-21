import os
from typing import Union, Dict, List
import contextlib
import joblib


def gen_data_config(data_root_path: str, save_dir_name: str,) -> List[Dict[str, Union[str, int]]]:
    data_configs = []
    root_folder_path = data_root_path + "/one_day_data"
    save_dir_path = data_root_path + f"/{save_dir_name}"

    for year in os.listdir(root_folder_path):
        for month in os.listdir(root_folder_path + f"/{year}"):
            for date in os.listdir(root_folder_path + f"/{year}/{month}"):
                if len(os.listdir(root_folder_path + f"/{year}/{month}/{date}")) > 0:
                    csv_file_names = os.listdir(root_folder_path + f"/{year}/{month}/{date}")
                    for csv_file_name in csv_file_names:
                        data_file_path = root_folder_path + f"/{year}/{month}/{date}/{csv_file_name}"
                        conf = {}
                        conf["data_file_path"] = data_file_path
                        conf["csv_file_name"] = csv_file_name
                        conf["save_dir_path"] = save_dir_path
                        conf["year"] = year
                        conf["month"] = month
                        conf["date"] = date
                        data_configs.append(conf)

    return data_configs


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwards):
            super().__init__(*args, **kwards)

        def __call__(self, *args, **kwards):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwards)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
