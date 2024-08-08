import sys
import os
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(file_list):
    ps, ns, fps, fns = [], [], [], []
    n_samples, n_adapt, n_target, n_boot = [], [], [], []
    for ff in file_list:
        if 'scores.csv' not in ff:
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        ps.append(np.sum(sheet['pre_ps']))
        ns.append(np.sum(sheet['pre_ns']))
        fps.append(np.sum(sheet['fps']))
        fns.append(np.sum(sheet['fns']))
        if 'n_al' in sheet.keys() and len(sheet)>1:
            n_adapt.append(np.sum(sheet['n_al']))
            if np.isnan(np.array(sheet['p_target'])[-1]):
                n_target.append(0)
            else:
                n_target.append(int(np.sum(np.array(sheet['p_target'])[-1]*np.sum(sheet['n_al']))))
        else:
            n_adapt.append(np.array(sheet['n_samples'])[-1])
            n_target.append(n_adapt[-1]*np.array(sheet['p_target'])[-1])
        n_boot.append(sheet['n_samples'][0]-sheet['n_al'][0])
        n_samples.append(np.sum(sheet['pre_ps'])+np.sum(sheet['pre_ns']))
    fnr = np.sum(fns)/np.sum(ps)
    fpr = np.sum(fps)/np.sum(ns)
    dcf = 0.25*fpr + 0.75*fnr
    if 'pml' in file_list[0]:
        n_adapt = [11625266]
    imlm = (np.sum(fps)+np.sum(n_adapt)+np.sum(n_boot))/np.sum(n_samples) + np.sum(fns)/np.sum(ps)
    p_adapt = np.sum(n_adapt)/np.sum(n_samples)*100
    p_target = np.sum(n_target)/np.sum(n_adapt)*100
    n_adapt = np.sum(n_adapt)+np.sum(n_boot)
    print('DCF & FNR & FPR & N_adapt & IMLM & % adapt & % targ')
    print(f'{dcf:.4f} & {fnr:.4f} & {fpr:.4f} & {n_adapt:d} & {imlm:.4f} & {p_adapt:.2f} & {p_target:.2f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1:])