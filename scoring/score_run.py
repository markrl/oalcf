import sys
import os
import numpy as np
import pandas as pd

def main(score_path):
    if 'scores.csv' not in score_path:
        score_file = os.path.join(score_path, 'scores.csv')
    else:
        score_file = score_path
    sheet = pd.read_csv(score_file)
    dcf = np.array(sheet['cum_dcf'])[-1]
    n_samples = np.array(sheet['n_samples'])[-1]
    perc_targ = np.array(sheet['p_target'])[-1]*100
    fps = np.sum(sheet['fps'])
    fns = np.sum(sheet['fns'])
    ns = np.sum(sheet['ns'])
    ps = np.sum(sheet['ps'])
    imlm = (fps+n_samples)/ns + fns/ps
    fnr = fns/ps
    fpr = fps/ns
    p_samples = n_samples/(ns+ps+n_samples)
    n_boot = sheet['n_samples'][0] - sheet['n_al'][0]
    n_al = np.sum(sheet['n_al'])
    n_cf_tp = np.sum(sheet['cf_tp'])
    n_cf_fp = np.sum(sheet['cf_fp'])
    print('DCF\tFNR\tFPR\tIMLM\t#bootstrap\t#AL\t#CF FP\t#CF TP\t% target in adapt pool\ttot labels/tot samples')
    print(f'{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{imlm:.4f}\t{n_boot:d}\t\t{n_al:d}\t{n_cf_fp:d}\t{n_cf_tp:d}\t{perc_targ:.2f}\t\t\t{p_samples:.4f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1])