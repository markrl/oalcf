import sys
import os
import numpy as np
import pandas as pd

def main(score_path):
    if 'tsd' in score_path:
        parts = score_path.split('_')
        spk = 'spk04'
        for ii,pp in enumerate(parts):
            if 'tsd' in pp:
                spk = parts[ii+1]
        label_root = os.path.join('/data/VTD/TSD_labels', spk+'_labels_1hr')
    elif 'tdd' in score_path:
        parts = score_path.split('_')
        dst = 'tv'
        for ii,pp in enumerate(parts):
            if 'tdd' in pp:
                dst = parts[ii+1]
        label_root = os.path.join('/data/VTD/TDD_labels', dst+'_labels_1hr')
    else:
        label_root = '/data/VTD/VTD_labels/all_labels_1hr'
    if 'scores.csv' not in score_path:
        score_file = os.path.join(score_path, 'scores.csv')
    else:
        score_file = score_path
    sheet = pd.read_csv(score_file)
    n_samples = np.array(sheet['n_samples'])[-1]
    perc_targ = np.array(sheet['p_target'])[-1]*100
    fps = np.sum(sheet['fps'])
    fns = np.sum(sheet['fns'])

    samples_path = os.path.join(os.path.split(score_file)[0], 'al_samples.txt')
    sample_sheet = pd.read_csv(samples_path, names=['file', 'idx'])
    ns, ps = 0, 0
    for ff in np.unique(sample_sheet['file']):
        rm,se,mc,hr = ff.split('_')
        label_file = os.path.join(label_root, '_'.join([rm,se,hr])+'.npy')
        labels = np.load(label_file)
        ps += np.sum(labels)
        ns += np.sum(1-labels)
    for ii,diff in enumerate(np.diff(sheet['n_samples'])):
        if diff==0:
            ns += sheet['ns'][ii+1]
            ps += sheet['ps'][ii+1]
    fnr = fns/ps
    fpr = fps/ns
    dcf = 0.25*fpr + 0.75*fnr
    p_samples = n_samples/(ns+ps)*100
    print('DCF\tFNR\tFPR\t% adapt\t% targ')
    print(f'{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{p_samples:.2f}\t{perc_targ:.2f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1])