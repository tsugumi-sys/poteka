import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tracemalloc

# LINE nortification
import requests
def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = '***'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

# check memory
def format_bytes(size):
    power = 2 ** 10
    n = 0
    power_labels = ["B", "KB", "MB", "GB", "TB"]
    while size > power and n <= len(power_labels):
        size /= power
        n += 1
    return "current used memory: {:.3f} {}".format(size, power_labels[n])

def log_memory():
    snapshot = tracemalloc.take_snapshot()
    size = sum([stat.size for stat in snapshot.statistics('filename')])
    print(format_bytes(size))


# Dataset Prepare
def makeData(n_samples, ymd, n_flames=6, data_type="train_data"):
    input_images = []
    label_images = []
    image_size = [int(436 * 0.25), int(257 * 0.25)] # height, width
    for i in range(1, divmod(n_samples, 12)[0]+1):
        row = []
        for j in range(1, 13):
            imageFileName = ymd + "-sample{}-image{}-ForStudy.png".format(i, j)
            #imageFilePath = "../../Cripped-Image-ForML/" + ymd + "/" + imageFileName
            imageFilePath = "../../Cripped-Image-ForML/" + data_type + "/" + ymd + "/" + imageFileName # only for Cripped-Image-ForML data
            if os.path.exists(imageFilePath):
                img = cv2.imread(imageFilePath)
                img2 = cv2.resize(img, (image_size[1], image_size[0]))
                row.append(img2 / 255.0)
            else:
                break
        if len(row) == n_flames * 2:
            input_images.append(row[:6])
            label_images.append(row[6:])
        else:
            continue
    input_row = np.array(input_images).reshape([len(input_images), 6, image_size[0], image_size[1], 3])
    label_row = np.array(label_images).reshape([len(label_images), 6, image_size[0], image_size[1], 3])
    return input_row, label_row

tracemalloc.start()
def ImageGenerator(imageFolderPath="../../Cripped-Image-ForML/", data_type="train_data"):
    count = 0
    imageFolderPath = imageFolderPath + data_type + "/" # change imageFolderPath
    for ymd in os.listdir(imageFolderPath):
        if "2020" in ymd:
            if len(os.listdir(imageFolderPath+ymd)) > 0:
                count += 1
                if count == 1:
                    input_set, label_set = makeData(n_samples=len(os.listdir(imageFolderPath+ymd)), ymd=ymd, data_type=data_type)
                else:
                    input_set_sec, label_set_sec = makeData(n_samples=len(os.listdir(imageFolderPath+ymd)), ymd=ymd, data_type=data_type)
                    input_set = np.append(input_set, input_set_sec, axis=0)
                    label_set = np.append(label_set, label_set_sec, axis=0)
            else:
                continue
            log_memory()
        else:
            continue
    return input_set, label_set


# check folder number
#folder_num = 0
#for x, y in ImageGenerator(imageFolderPath="../../Cripped-Image-ForML/validation_data/", data_type="validation_data"):
#    folder_num += x.shape[0]
#    print("Input Shape: {}  Label Shape: {}".format(x.shape, y.shape))
#    print(folder_num)

# ML Code
image_size = [int(436 * 0.25), int(257 * 0.25)] # height, width
seq = keras.Sequential(
    [
        keras.Input(
            shape=(None, image_size[0], image_size[1], 3)
        ),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu"
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu"
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu"
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=3, kernel_size=(3, 3, 3), padding="same"
        ),
    ]
)
# optimizer="adadelta"
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
seq.compile(loss="mean_absolute_error", optimizer=optimizer, metrics=['acc'])
seq.summary()

# validation data:  ImageGenerator(imageFolderPath="../../Cripped-Image-ForML/validation_data", data_type="validation_data")
# train data: ImageGenerator(imageFolderPath="../../Cripped-Image-ForML/train_data")
# learning data

input_data, label_data = ImageGenerator()
print(input_data.shape)
print(label_data.shape)
try:
    history = seq.fit(
    input_data,
    label_data,
    #steps_per_epoch=1,
    epochs=500,
    batch_size=16,
    verbose=1,
    validation_split=0.1
    #validation_data=ImageGenerator(imageFolderPath="../../testItem/test-convlstm/validation_data/", data_type="validation_data"),
    )
    model_name = "test-model5-10"
    if not os.path.exists("./"+model_name):
        os.mkdir("./"+model_name)

    # save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('./'+model_name+'/history.csv')
    print("history saved!!!")

    # save model
    seq.save('./'+model_name+'/test_model.h5')
    print('model saved!!!')
    del seq
    send_line_notify("Process has successfully ended!!!")
except:
    import traceback
    send_line_notify("Process has Stopped with some error!!!")
    send_line_notify(traceback.format_exc())
    print(traceback.format_exc())