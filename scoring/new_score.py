import sys
import os
import numpy as np
import pandas as pd

from pdb import set_trace

def main(score_paths):
    score_paths.sort()
    ns, ps = 0, 0
    pre_fns, pre_fps = 0, 0
    fns, fps = 0, 0
    cf_fns, cf_fps = 0, 0
    diag_fns, diag_fps = 0, 0
    n_al = 0
    print('\tDCF\tFNR\tFPR\tIMLM')
    for score_path in score_paths:
        if '.csv' not in score_path:
            score_file = os.path.join(score_path, 'scores.csv')
        else:
            score_file = score_path
        sheet = pd.read_csv(score_file)
        ns += np.sum(sheet['pre_ns'])
        ps += np.sum(sheet['pre_ps'])
        pre_fps += np.sum(sheet['pre_fps'])
        pre_fns += np.sum(sheet['pre_fns'])
        fps += np.sum(sheet['fps'])
        fns += np.sum(sheet['fns'])
        cf_fps += np.sum(sheet['cf_fps'])
        cf_fns += np.sum(sheet['cf_fns'])
        diag_fps += np.sum(sheet['diag_fps'])
        diag_fns += np.sum(sheet['diag_fns'])
        n_boot = sheet['n_samples'][0] - sheet['n_al'][0] - sheet['cf_tp'][0] - sheet['cf_fp'][0]
        n_al += np.sum(sheet['n_al']) + n_boot
    pre_fnr = pre_fns/ps
    pre_fpr = pre_fps/ns
    pre_dcf = 0.75*pre_fnr + 0.25*pre_fpr
    pre_imlm = pre_fps/(ps+ns) + pre_fns/ps
    fnr = fns/ps
    fpr = fps/ns
    dcf = 0.75*fnr + 0.25*fpr
    imlm = (fps+n_al)/(ps+ns) + fns/ps
    cf_fnr = cf_fns/ps
    cf_fpr = cf_fps/ns
    cf_dcf = 0.75*cf_fnr + 0.25*cf_fpr
    cf_imlm = (cf_fps+n_al)/(ps+ns) + cf_fns/ps
    diag_fnr = diag_fns/ps
    diag_fpr = diag_fps/ns
    diag_dcf = 0.75*diag_fnr + 0.25*diag_fpr
    diag_imlm = (diag_fps+n_al)/(ps+ns) + diag_fns/ps
    print(f'Pre\t{pre_dcf:.4f}\t{pre_fnr:.4f}\t{pre_fpr:.4f}\t{pre_imlm:.4f}')
    print(f'Post\t{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{imlm:.4f}')
    print(f'CF\t{cf_dcf:.4f}\t{cf_fnr:.4f}\t{cf_fpr:.4f}\t{cf_imlm:.4f}')
    print(f'Diag\t{diag_dcf:.4f}\t{diag_fnr:.4f}\t{diag_fpr:.4f}\t{diag_imlm:.4f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1:])