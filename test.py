import csv
import re

def lonlat(filenum):
    regex = re.compile('\d+')
    latlon = []


    with open('observationpoint'+str(filenum)+'.csv') as f:
        reader = csv.reader(f)
        l = [row for row in reader]

        for row2 in l:
            becon = []
            try:
              lat = float(regex.findall(row2[2])[0])
              lon = float(regex.findall(row2[3])[0])
              becon.append(lat / 10000)
              becon.append(lon / 10000)
              latlon.append(becon)
            except IndexError:
                continue

    print('Open file and extract lontitude and latitude')

    uniqued = []
    for item in latlon:
        if not item in uniqued:
            uniqued.append(item)
    print(uniqued)

    print('making csv file!!')

    with open('./latlon'+str(filenum)+'.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(uniqued)

for i in range(15, 40):
   lonlat(i)