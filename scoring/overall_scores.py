import sys
import os
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(dir_code):
    ps, ns, fps, fns = [], [], [], []
    n_samples, n_adapt, n_target = [], [], []
    dir_components = dir_code.split('_')
    dirs = ['_'.join(dir_components + ['vtd']), '_'.join(dir_components + ['lid'])]
    file_list = glob.glob(os.path.join('output', dirs[0], '*')) + glob.glob(os.path.join('output', dirs[1], '*'))
    for ff in file_list:
        if 'scores.csv' not in ff:
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        ps.append(np.sum(sheet['ps']))
        ns.append(np.sum(sheet['ns']))
        fps.append(np.sum(sheet['fps']))
        fns.append(np.sum(sheet['fns']))
        if 'n_al' in sheet.keys() and np.array(sheet['n_al'])[-1]>0:
            n_adapt.append(np.sum(sheet['n_al']))
            n_target.append(int(np.sum(sheet['p_target']*sheet['n_al'])))
        else:
            n_adapt.append(np.array(sheet['n_samples'])[-1])
            n_target.append(n_adapt[-1]*np.array(sheet['p_target'])[-1])
        n_samples.append(ps[-1]+ns[-1])
    fnr = np.sum(fns)/np.sum(ps)
    fpr = np.sum(fps)/np.sum(ns)
    dcf = 0.25*fpr + 0.75*fnr
    imlm = (np.sum(fps)+np.sum(n_adapt))/(np.sum(ns)+np.sum(ps)) + np.sum(fns)/np.sum(ps)
    p_adapt = np.sum(n_adapt)/np.sum(n_samples)*100
    p_target = np.sum(n_target)/np.sum(n_adapt)*100
    print('DCF\tFNR\tFPR\tIMLM\t% adapt\t% targ')
    print(f'{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{imlm:.4f}\t{p_adapt:.2f}\t{p_target:.2f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1])