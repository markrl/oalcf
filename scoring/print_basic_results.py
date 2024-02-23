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
    p_samples = n_samples/(ns+ps+n_samples)*100
    print('DCF\tFNR\tFPR\t% adapt\t% targ\tIMLM')
    print(f'{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{p_samples:.2f}\t{perc_targ:.2f}\t{imlm:.4f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1])