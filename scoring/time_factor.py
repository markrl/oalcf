import os
import glob
import sys
import pandas as pd
import numpy as np

from pdb import set_trace

def main(rootdir):
    csv_list = glob.glob(os.path.join(rootdir, '*', 'scores.csv'))
    txt_list = glob.glob(os.path.join(rootdir, '*', 'time.txt'))
    csv_list.sort()
    txt_list.sort()

    csv_times, data_times, txt_times = [], [], []
    for csv,txt in zip(csv_list,txt_list):
        sheet = pd.read_csv(csv)
        csv_times.append(np.sum(sheet['time'])) # seconds
        data_times.append(len(sheet)) # hours
        txt_times.append(float(np.genfromtxt(txt))) # seconds
    
    difference_s = np.sum(txt_times) - np.sum(csv_times)
    print(f'Difference between total and loop only: {seconds2hms(difference_s)}')
    factor = np.sum(csv_times)/(np.sum(data_times)*3600)   
    print(f'RT factor: {factor:.4f}')
          
def seconds2hms(secs):
    h = int(secs // 3600)
    secs -= 3600*h
    m = int(secs//60)
    s = secs - m*60
    return f'{h:2d}:{m:02d}:{s:.2f}'

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1])
