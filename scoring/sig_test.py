from scipy.stats import ttest_rel
import os
import sys
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(run1, run2, prefix=''):
    paths1 = []
    for dd in run1.split(','):
        paths1 += glob.glob(os.path.join(dd, '*', 'scores.csv'))
    paths1.sort()

    paths2 = []
    for dd in run2.split(','):
        paths2 += glob.glob(os.path.join(dd, '*', 'scores.csv'))
    paths2.sort()

    dcfs1, fnrs1, fprs1, imlms1 = [], [], [], []
    for pp in paths1:
        sheet = pd.read_csv(pp)
        dcfs1.append(np.array(sheet[prefix+'cum_dcf'])[-1])
        n_samples = np.array(sheet['n_samples'])[-1]
        fns = np.sum(sheet[prefix+'fns'])
        fps = np.sum(sheet[prefix+'fps'])
        ns = np.sum(sheet[prefix+'ns'])
        ps = np.sum(sheet[prefix+'ps'])
        if ps==0:
            fnrs1.append(0)
            imlms1.append((fps+n_samples)/ns)
            fprs1.append(fps/ns)
        elif ns==0:
            fnrs1.append(fns/ps)
            imlms1.append(fns/ps)
            fprs1.append(0)
        else:
            fnrs1.append(fns/ps)
            fprs1.append(fps/ns)
            imlms1.append((fps+n_samples)/ns + fns/ps)

    dcfs2, fnrs2, fprs2, imlms2 = [], [], [], []
    for pp in paths2:
        sheet = pd.read_csv(pp)
        dcfs2.append(np.array(sheet[prefix+'cum_dcf'])[-1])
        n_samples = np.array(sheet['n_samples'])[-1]
        fns = np.sum(sheet[prefix+'fns'])
        fps = np.sum(sheet[prefix+'fps'])
        ns = np.sum(sheet[prefix+'ns'])
        ps = np.sum(sheet[prefix+'ps'])
        if ps==0:
            fnrs2.append(0)
            imlms2.append((fps+n_samples)/ns)
            fprs2.append(fps/ns)
        elif ns==0:
            fnrs2.append(fns/ps)
            imlms2.append(fns/ps)
            fprs2.append(0)
        else:
            fnrs2.append(fns/ps)
            fprs2.append(fps/ns)
            imlms2.append((fps+n_samples)/ns + fns/ps)

    dcf_pval = ttest_rel(dcfs1, dcfs2, alternative='less').pvalue
    fnr_pval = ttest_rel(fnrs1, fnrs2, alternative='less').pvalue
    fpr_pval = ttest_rel(fprs1, fprs2, alternative='less').pvalue
    imlm_pval = ttest_rel(imlms1, imlms2, alternative='less').pvalue
    print('DCF\tFNR\tFPR\tIMLM')
    print(f'{dcf_pval:.4f}\t{fnr_pval:.4f}\t{fpr_pval:.4f}\t{imlm_pval:.4f}')


if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    main(sys.argv[1], sys.argv[2])