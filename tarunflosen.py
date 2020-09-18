import os
import tarfile
import csv
import re
import traceback


def unflosenTar(path, files):
    for file in files: 
        with tarfile.open(path + '/'+ file , 'r:gz') as tar:
            tar.getmembers()
            tar.extractall(path = './' + file.replace('.tar.gz', ''))
            tar.close()
    return print('file saved')
#unflose tarFile in current directory


def extractLatlon(filepath):
    hk = 'HK'
    hkfile = os.listdir(filepath + '/' + hk)
    with open(filepath + '/' + hk + '/' + hkfile[0]) as f:
       reader = csv.reader(f)
       l = [row for row in reader]
       f.close()
    
    item = l[100]
    regex = re.compile('\d+')
    latlon = []
    lat = float(regex.findall(item[2])[0])
    lon = float(regex.findall(item[3])[0])
    latlon.append(lat / 10000)
    latlon.append(lon / 10000)
    return latlon



def extractWeatherdata(filepath, dataname):
    regex = re.compile('\d+')
    data = []
    time = []
    a = []

    datalist = ['', '', 'raf', 'ra1', 'ri1', 'era', 'cra', 'at1', 'rh1', 'prs', 'slp', 'wd1', 'wdm', 'ws1', 'wsm', 'wnd', 'wns', 'sol', 'wet', 'wbg', 'wea']
    magnilist = [1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 10, 10, 1, 10, 1, 1, 10, 1]

    dataIndex = datalist.index(dataname)
    magni = magnilist[dataIndex]

    weather = 'Weather'
    weatherfile = os.listdir(filepath + '/' + weather)
    with open(filepath + '/' + weather + '/' + weatherfile[0]) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        f.close()

    for item in l:
        try: 
            k = float(regex.findall(item[dataIndex])[1]) / magni
            data.append(k)
            time.append(item[1])

        except IndexError:
            continue
    
    a.append(time)
    a.append(data)
    return a



def makecsvfile(filename, data, latlon, month, date):
    with open('./2020-' + month + '-' + date + 'data/' + filename + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(latlon)
        for i in range(len(data[0])):
            row = []
            row.append(data[0][i])
            row.append(data[1][i])
            writer.writerow(row)
        f.close()



#cd desktop/temporary/pythoncodeforstudy/onedaydata

month = input('Month:   ')
date = input('Date: ')
PATH = './9-1'
files = os.listdir(PATH)
#unflosenTar(PATH, files)
for file in files:
    try:
        filename = file.replace('.tar.gz', '')
        filePATH = './' + filename + '/2020-09-01'
        data = extractWeatherdata(filePATH, 'at1')
        latlon = extractLatlon(filePATH)
        makecsvfile(filename, data, latlon, month, date)
        print(filename + ' successefully saved')
    except:
        print('error ocuured')
        print(traceback.format_exc())
        continue
