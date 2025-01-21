from scipy.stats import ttest_rel, wilcoxon
import os
import sys
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(run1, run2, prefix=''):
    paths1 = []
    for dd in run1.split(','):
        if 'extra' in dd:
            paths1 += glob.glob(os.path.join(dd, '*.csv'))
        else:
            paths1 += glob.glob(os.path.join(dd, '*', 'scores.csv'))
    paths1.sort()

    paths2 = []
    for dd in run2.split(','):
        if 'extra' in dd:
            paths2 += glob.glob(os.path.join(dd, '*.csv'))
        else:
            paths2 += glob.glob(os.path.join(dd, '*', 'scores.csv'))
    paths2.sort()

    n_boot = 8
    dcfs1, fnrs1, fprs1, imlms1 = [], [], [], []
    for pp in paths1:
        sheet = pd.read_csv(pp)
        n_samples = n_boot + np.sum(sheet['n_al'])
        fns = np.sum(sheet[prefix+'fns'])
        fps = np.sum(sheet[prefix+'fps'])
        # n_samples += np.sum(sheet['fps'])
        # fns = np.sum(sheet[prefix+'diag_fns'])
        # fps = np.sum(sheet[prefix+'diag_fps'])
        ns = np.sum(sheet['pre_ns'])
        ps = np.sum(sheet['pre_ps'])
        if ps==0:
            fnrs1.append(0)
            imlms1.append((fps+n_samples)/ns)
            fprs1.append(fps/ns)
            dcfs1.append(0.25*fps/ns)
        elif ns==0:
            fnrs1.append(fns/ps)
            imlms1.append(fns/ps)
            fprs1.append(0)
            dcfs1.append(0.75*fns/ps)
        else:
            fnrs1.append(fns/ps)
            fprs1.append(fps/ns)
            imlms1.append((fps+n_samples)/(ns+ps) + fns/ps)
            dcfs1.append(0.75*fns/ps + 0.25*fps/ns)

    dcfs2, fnrs2, fprs2, imlms2 = [], [], [], []
    for pp in paths2:
        sheet = pd.read_csv(pp)
        n_samples = n_boot + np.sum(sheet['n_al'])
        fns = np.sum(sheet[prefix+'fns'])
        fps = np.sum(sheet[prefix+'fps'])
        ns = np.sum(sheet['pre_ns'])
        ps = np.sum(sheet['pre_ps'])
        if ps==0:
            fnrs2.append(0)
            imlms2.append((fps+n_samples)/ns)
            fprs2.append(fps/ns)
            dcfs2.append(0.25*fps/ns)
        elif ns==0:
            fnrs2.append(fns/ps)
            imlms2.append(fns/ps)
            fprs2.append(0)
            dcfs2.append(0.75*fns/ps)
        else:
            fnrs2.append(fns/ps)
            fprs2.append(fps/ns)
            imlms2.append((fps+n_samples)/(ns+ps) + fns/ps)
            dcfs2.append(0.75*fns/ps + 0.25*fps/ns)

    dcf_pval_ttest = ttest_rel(dcfs1, dcfs2, alternative='less').pvalue
    fnr_pval_ttest = ttest_rel(fnrs1, fnrs2, alternative='less').pvalue
    fpr_pval_ttest = ttest_rel(fprs1, fprs2, alternative='less').pvalue
    imlm_pval_ttest = ttest_rel(imlms1, imlms2, alternative='less').pvalue

    dcf_pval_wilcoxon = wilcoxon(dcfs1, dcfs2, alternative='less', method='approx').pvalue
    fnr_pval_wilcoxon = wilcoxon(fnrs1, fnrs2, alternative='less', method='approx').pvalue
    fpr_pval_wilcoxon = wilcoxon(fprs1, fprs2, alternative='less', method='approx').pvalue
    imlm_pval_wilcoxon = wilcoxon(imlms1, imlms2, alternative='less', method='approx').pvalue
    print('\t\tDCF\tFNR\tFPR\tIMLM')
    print(f'T-test\t\t{dcf_pval_ttest:.4f}\t{fnr_pval_ttest:.4f}\t{fpr_pval_ttest:.4f}\t{imlm_pval_ttest:.4f}')
    print(f'Wilcoxon\t{dcf_pval_wilcoxon:.4f}\t{fnr_pval_wilcoxon:.4f}\t{fpr_pval_wilcoxon:.4f}\t{imlm_pval_wilcoxon:.4f}')


if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    main(sys.argv[1], sys.argv[2])