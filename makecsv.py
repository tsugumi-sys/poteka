import csv 
import os
import pandas as pd

month = input('month: ')
date = input('date: ')
path = './2020-' + month + '-' + date + 'ra1data'
files = os.listdir(path)
lon = []
lat = []
data = []
for file in files:
    with open(path + '/' + file) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        lon.append(l[0][0])
        lat.append(l[0][1])
        data.append(l[4][1])
        f.close()

df = pd.DataFrame({
    'rain': data,
    'longitude': lon,
    'latitude':  lat,
})
df.to_csv(path + '/' + 'rain.csv')

#cd desktop/temporary/pythoncodeforstudy/onedaydata