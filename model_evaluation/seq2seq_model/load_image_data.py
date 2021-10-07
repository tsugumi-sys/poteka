# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime, timedelta
# import tracemalloc
# import tensorflow as tf
# from PIL import Image

# def format_bytes(size):
#     power = 2 ** 10
#     n = 0
#     power_labels = ['B', 'KB', 'MB', 'GB', 'TB']
#     while size > power and n <= len(power_labels):
#         size /= power
#         n += 1
#     return f"current used memory: {size} {power_labels[n]}"

# def log_memory():
#     snapshot = tracemalloc.take_snapshot()
#     size = sum([stat.size for stat in snapshot.statistics('filename')])
#     print(format_bytes(size))


# # return: ndarray
# def load_image_data(path: str, img_height=75, img_width=45):
#     image = Image.open(path)
#     image = image.resize((img_width, img_height))
#     image = image.convert('RGB')
#     return np.asarray(image) / 255 #(np.asarray(image) - 128)/128

# def datetime_range(start, end, delta):
#     current = start
#     while current <= end:
#         yield current
#         current += delta

# def create_time_list(year=2020, month=1, date=1):
#     dts = [dt for dt in datetime_range(datetime(year, month, date, 0), datetime(year, month, date, 23, 59), timedelta(minutes=10))]
#     return dts


# def load_rainbow_rain_data(img_height=60, img_width=36):
#     tracemalloc.start()

#     time_list = create_time_list()

#     input_arr = []
#     label_arr = []

#     year = 2019
#     monthes = ['10', '11']
#     dates_list = []
#     time_set = []
#     for month in monthes:
#         log_memory()
#         dates = os.listdir(f'../../data/train_with_image/rain_image/{year}/{month}')
#         dates_list.append(dates)

#         for date in dates:
#             print(date)
#             rain_path = f'../../data/train_with_image/rain_image/{year}/{month}/{date}'
#             time_subset = []
#             if os.path.exists(rain_path):
#                 rain_file_num = len(os.listdir(rain_path))
#                 if rain_file_num == 288:
#                     for step in range(0, len(time_list) - 6, 6):
#                         file_names = [f'{dt.hour}-{dt.minute}croped.png' for dt in time_list[step:step+12]]

#                         time_subset.append(file_names[6:])

#                         subset_arrs = []
#                         for file_name in file_names:
#                             # Load data
#                             rain_file_path = rain_path + f'/{file_name}'

#                             # Create ndarray
#                             img_arr = load_image_data(rain_file_path,img_height=img_height, img_width=img_width)

#                             subset_arrs.append(img_arr)

#                         input_arr.append(subset_arrs[:6])
#                         label_arr.append(subset_arrs[6:])
#             time_set.append(time_subset)

#     input_arr = np.array(input_arr).reshape([len(input_arr), 6, img_height, img_width, 3])
#     label_arr = np.array(label_arr).reshape([len(label_arr), 6, img_height, img_width, 3])

#     data_config = {
#         'year': year,
#         'monthes': monthes,
#         'dates': dates_list,
#         'time': time_set
#     }

#     return input_arr, label_arr, data_config


# def load_dense_rain_data(img_height=60, img_width=36):
#     tracemalloc.start()

#     time_list = create_time_list()

#     input_arr = []
#     label_arr = []

#     year = 2019
#     monthes = ['10', '11']
#     dates_list = []
#     time_set = []
#     for month in monthes:
#         log_memory()
#         dates = os.listdir(f'../../data/train_with_image/dense_rain_image/{year}/{month}')
#         dates_list.append(dates)

#         for date in dates:
#             print(date)
#             rain_path = f'../../data/train_with_image/dense_rain_image/{year}/{month}/{date}'
#             time_subset = []
#             if os.path.exists(rain_path):
#                 rain_file_num = len(os.listdir(rain_path))
#                 if rain_file_num == 288:
#                     for step in range(0, len(time_list) - 6, 6):
#                         file_names = [f'{dt.hour}-{dt.minute}croped.png' for dt in time_list[step:step+12]]
#                         time_subset.append(file_names[6:])

#                         subset_arrs = []
#                         for file_name in file_names:
#                             # Load data
#                             rain_file_path = rain_path + f'/{file_name}'

#                             # Create ndarray
#                             img_arr = load_image_data(path=rain_file_path, img_height=img_height, img_width=img_width)

#                             subset_arrs.append(img_arr)

#                         input_arr.append(subset_arrs[:6])
#                         label_arr.append(subset_arrs[6:])
#             time_set.append(time_subset)

#     input_arr = np.array(input_arr).reshape([len(input_arr), 6, img_height, img_width, 3])
#     label_arr = np.array(label_arr).reshape([len(label_arr), 6, img_height, img_width, 3])

#     data_config = {
#         'year': year,
#         'monthes': monthes,
#         'dates': dates_list,
#         'time': time_set
#     }

#     return input_arr, label_arr, data_config


# if __name__ == '__main__':
#     load_data()
