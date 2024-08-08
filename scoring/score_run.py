import sys
import os
import numpy as np
import pandas as pd

from pdb import set_trace

def main(score_paths):
    score_paths.sort()
    print('DCF\tFNR\tFPR\tIMLM\t#bootstrap\t#AL\t#FP\t#FN\t#Target\t#Nontarget\t#CF retained\t% target in adapt pool\ttot labels/tot samples')
    for score_path in score_paths:
        if '.csv' not in score_path:
            score_file = os.path.join(score_path, 'scores.csv')
        else:
            score_file = score_path
        sheet = pd.read_csv(score_file)
        n_samples = np.array(sheet['n_samples'])[-1]
        perc_targ = np.array(sheet['p_target'])[-1]*100
        fps = np.sum(sheet['fps'])
        fns = np.sum(sheet['fns'])
        ns = np.sum(sheet['pre_ns'])
        ps = np.sum(sheet['pre_ps'])
        fnr = fns/ps
        fpr = fps/ns
        dcf = 0.75*fnr + 0.25*fpr
        n_boot = sheet['n_samples'][0] - sheet['n_al'][0] - sheet['cf_tp'][0] - sheet['cf_fp'][0]
        n_al = np.sum(sheet['n_al'])
        imlm = (fps+n_boot+n_al)/(ps+ns) + fns/ps
        n_cf_tp = np.sum(sheet['cf_tp'])
        n_cf_fp = np.sum(sheet['cf_fp'])
        n_cf = n_cf_tp + n_cf_fp
        p_samples = (n_al+fps+n_boot)/(ns+ps)
        print(f'{dcf:.4f}\t{fnr:.4f}\t{fpr:.4f}\t{imlm:.4f}\t{n_boot:d}\t\t{n_al:d}\t{fps:d}\t{fns:d}\t{ps:d}\t{ns:d}\t\t{n_cf:d}\t\t{perc_targ:.2f}\t\t\t{p_samples:.4f}')

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    main(sys.argv[1:])