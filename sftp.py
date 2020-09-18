import os 
import paramiko
import traceback
import time


def getFile(year_date):
    USER = 'ulat'
    HOST = 'www.ep.sci.hokudai.ac.jp'
    PORT = 22
    PSWD = 't7P50k%A2'

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    client.connect(HOST, port=PORT, username=USER, password=PSWD)

    try:
        sftp_connection = client.open_sftp()
        sftp_connection.chdir('P-POTEKA')
        entries = sftp_connection.listdir()
    #   len(entries) = 36
        for e in entries:
            filename = e
            sftp_connection.chdir(filename)
            if (year_date + '.tar.gz' in sftp_connection.listdir()):
                sftp_connection.get( year_date + '.tar.gz', filename+'.tar.gz')
                print('file ' + e + ' downloaded')
                time.sleep(1)
            else:
                print('There is not ' + year_date + '.tar.gz file in ' + filename + 'Directory')
            sftp_connection.chdir('..')

    except:
        print('Erorr!!!!!!')
        print(traceback.format_exc())


    finally:
        client.close()

# sftp_connection.get('sample.txt', 'sample.txt')

year = input('Enter year: ')
month = input('Enter month: ')
date = input('Enter date: ')
year_date = year + '-' + month + '-' + date
print('-------------------------------------')
print('fecthing files ')
getFile(year_date)

#cd desktop/temporary/pythoncodeforstudy/onedaydata