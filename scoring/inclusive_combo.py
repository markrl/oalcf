import sys
import os
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(file_list):
    if 'tsd' in file_list[0]:
        parts = file_list[0].split('_')
        spk = 'spk04'
        for ii,pp in enumerate(parts):
            if 'tsd' in pp:
                spk = parts[ii+1]
        label_root = os.path.join('/data/VTD/TSD_labels', spk+'_labels_1hr')
    elif 'tdd' in file_list[0]:
        parts = file_list[0].split('_')
        dst = 'tv'
        for ii,pp in enumerate(parts):
            if 'tdd' in pp:
                dst = parts[ii+1]
        label_root = os.path.join('/data/VTD/TDD_labels', dst+'_labels_1hr')
    else:
        label_root = '/data/VTD/VTD_labels/all_labels_1hr'
    ps, ns, fps, fns = [], [], [], []
    n_samples, n_adapt, n_target = [], [], []
    for ff in file_list:
        if 'scores.csv' not in ff:
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        # fps.append(np.sum(sheet['pre_fps']))
        # fns.append(np.sum(sheet['pre_fns']))
        fps.append(np.sum(sheet['fps']))
        fns.append(np.sum(sheet['fns']))
        n_adapt.append(np.array(sheet['n_samples'])[-1])
        n_target.append(n_adapt[-1]*np.array(sheet['p_target'])[-1])

        samples_path = os.path.join(os.path.split(ff)[0], 'al_samples.txt')
        sample_sheet = pd.read_csv(samples_path, names=['file', 'idx'])
        current_ps, current_ns = 0, 0
        for adapt_file in np.unique(sample_sheet['file']):
            rm,se,mc,hr = adapt_file.split('_')
            label_file = os.path.join(label_root, '_'.join([rm,se,hr])+'.npy')
            labels = np.load(label_file)
            current_ps += np.sum(labels)
            current_ns += np.sum(1-labels)
        for ii,diff in enumerate(np.diff(sheet['n_samples'])):
            if diff==0:
                current_ns += sheet['ns'][ii+1]
                current_ps += sheet['ps'][ii+1]
        ps.append(current_ps)
        ns.append(current_ns)
        n_samples.append(ps[-1]+ns[-1])
    fnr = np.sum(fns)/np.sum(ps)
    fpr = np.sum(fps)/np.sum(ns)
    dcf = 0.25*fpr + 0.75*fnr
    imlm = (np.sum(fps)+np.sum(n_adapt))/(np.sum(ns)+np.sum(ps)) + np.sum(fns)/np.sum(ps)
    p_adapt = np.sum(n_adapt)/np.sum(n_samples)*100
    p_target = np.sum(n_target)/np.sum(n_adapt)*100
    print('DCF\tFNR\tFPR\t% adapt\t% targ\tIMLM')
    print(f'{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{p_adapt:.2f}\t{p_target:.2f}\t{imlm:.4f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1:])