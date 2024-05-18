import sys
import os
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(file_list):
    n_samples, n_cf, n_al, n_target = [], [], [], []
    for ff in file_list:
        if 'scores.csv' not in ff:
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        if 'n_cf' in sheet.keys():
            n_cf.append(np.sum(sheet['n_cf']))
        else:
            n_cf.append(0)
        if 'n_al' in sheet.keys():
            n_al.append(np.sum(sheet['n_al']))
        else:
            n_al.append(np.array(sheet['n_samples'])[-1])
        n_samples.append(n_al[-1]+np.sum(sheet['ns'])+np.sum(sheet['ps']))
        n_target.append(np.array(sheet['p_target'])[-1]*(n_al[-1] + n_cf[-1]))

    n_samples = np.sum(n_samples)
    n_al = np.sum(n_al)
    n_cf = np.sum(n_cf)
    n_adapt = n_al + n_cf
    p_target = np.sum(n_target)/n_adapt*100
    print('#samp\t#AL\t#CF\t#adapt\t%targ')
    print(f'{n_samples:d}\t{n_al:d}\t{n_cf:d}\t{n_adapt:d}\t{p_target:.2f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1:])