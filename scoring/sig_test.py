from scipy.stats import ttest_rel, wilcoxon
import os
import sys
import glob
import numpy as np
import pandas as pd

from pdb import set_trace

def main(run1, run2, prefix=''):
    TRAIN_SIZES = {'apartment_mc19': 835200,
                   'hotel_mc19': 835200,
                   'office_mc13': 835200,
                   'rm1_mc20':1300320,
                   'rm2_mc16':1300320,
                   'rm3_mc16':1300320,
                   'rm4_mc20':1300320,
                   'bas':982800,
                   'ckb':666720,
                   'cnh':797040,
                   'cv':666720,
                   'dv':797040,
                   'ha':982800,
                   'hi':797040,
                   'hy-AM':666720,
                   'id':1363680,
                   'kmr':666720,
                   'ky':666720,
                   'ml':797040,
                   'mr':797040,
                   'or':797040,
                   'skr':797040,
                   'sr':666720,
                   'tt':666720,
                   'vi':1363680,
                   'yo':982800,
                   'yue':1363680}
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

    dcfs1, fnrs1, fprs1, imlms1 = [], [], [], []
    for pp in paths1:
        sheet = pd.read_csv(pp)
        if 'pml' in pp:
            env = pp.split('/')[-2]
            n_boot = TRAIN_SIZES[env]
        else:
            n_boot = 8
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
        if 'pml' in pp:
            env = pp.split('/')[-2]
            n_boot = TRAIN_SIZES[env]
        else:
            n_boot = 8
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
    print(f'T-test\t\t{dcf_pval_ttest:.1e}\t{fnr_pval_ttest:.1e}\t{fpr_pval_ttest:.1e}\t{imlm_pval_ttest:.1e}')
    print(f'Wilcoxon\t{dcf_pval_wilcoxon:.1e}\t{fnr_pval_wilcoxon:.1e}\t{fpr_pval_wilcoxon:.1e}\t{imlm_pval_wilcoxon:.1e}')


if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    main(sys.argv[1], sys.argv[2])