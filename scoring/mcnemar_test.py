import os
import sys
import glob
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from pdb import set_trace

def main(run1, run2):
    lang_lengths = {'bas':56880,
                    'ckb':114480,
                    'cnh':68400,
                    'cv':114480,
                    'dv':68400,
                    'ha':56880,
                    'hi':68400,
                    'hy-AM':114480,
                    'id':90720,
                    'kmr':114480,
                    'ky':114480,
                    'ml':68400,
                    'mr':68400,
                    'or':68400,
                    'skr':68400,
                    'sr':114480,
                    'tt':114480,
                    'vi':90720,
                    'yo':56880,
                    'yue':90720,
                    'apartment':93600,
                    'hotel':92160,
                    'office':92880,
                    'rm1':42480,
                    'rm2':68400,
                    'rm3':54720,
                    'rm4':56160,}

    fns1, fps1 = [], []
    for dd in run1.split(','):
        fns1 += glob.glob(os.path.join(dd, '*', 'fn_list.txt'))
        fps1 += glob.glob(os.path.join(dd, '*', 'fp_list.txt'))
    fns1.sort()
    fps1.sort()

    fns2, fps2 = [], []
    for dd in run2.split(','):
        fns2 += glob.glob(os.path.join(dd, '*', 'fn_list.txt'))
        fps2 += glob.glob(os.path.join(dd, '*', 'fp_list.txt'))
    fns2.sort()
    fps2.sort()

    table = np.zeros((2,2))
    for fn1,fn2,fp1,fp2 in zip(fns1,fns2,fps1,fps2):
        lang = fn1.split('/')[-2]
        lang = lang.split('_')[0]
        errors1 = set(np.genfromtxt(fn1, dtype=int)).union(set(np.genfromtxt(fp1, dtype=int)))
        correct1 = set(np.arange(lang_lengths[lang])).difference(errors1)
        errors2 = set(np.genfromtxt(fn2, dtype=int)).union(set(np.genfromtxt(fp2, dtype=int)))
        correct2 = set(np.arange(lang_lengths[lang])).difference(errors2)

        # Top left: Both correct
        table[0,0] += len(correct1.intersection(correct2))
        # Top right: trial 2 incorrect, trial 1 correct
        table[0,1] += len(correct1.intersection(errors2))
        # Bottom left: trial 2 correct, trial 1 incorrect
        table[1,0] += len(errors1.intersection(correct2))
        # Bottom right: Both incorrect
        table[1,1] += len(errors1.intersection(errors2))
    p_val = mcnemar(table).pvalue
    print('OVERALL')
    print(f'{p_val:.1e}')
    print(table)
    print()

    table = np.zeros((2,2))
    for fn1,fn2 in zip(fns1,fns2):
        lang = fn1.split('/')[-2]
        lang = lang.split('_')[0]
        errors1 = set(np.genfromtxt(fn1, dtype=int))
        correct1 = set(np.arange(lang_lengths[lang])).difference(errors1)
        errors2 = set(np.genfromtxt(fn2, dtype=int))
        correct2 = set(np.arange(lang_lengths[lang])).difference(errors2)

        # Top left: Both correct
        table[0,0] += len(correct1.intersection(correct2))
        # Top right: trial 2 incorrect, trial 1 correct
        table[0,1] += len(correct1.intersection(errors2))
        # Bottom left: trial 2 correct, trial 1 incorrect
        table[1,0] += len(errors1.intersection(correct2))
        # Bottom right: Both incorrect
        table[1,1] += len(errors1.intersection(errors2))
    p_val_sensitivity = mcnemar(table).pvalue
    print('SENSITIVITY')
    print(f'{p_val_sensitivity:.1e}')
    print(table)
    print()

    table = np.zeros((2,2))
    for fp1,fp2 in zip(fps1,fps2):
        lang = fp1.split('/')[-2]
        lang = lang.split('_')[0]
        errors1 = set(np.genfromtxt(fp1, dtype=int))
        correct1 = set(np.arange(lang_lengths[lang])).difference(errors1)
        errors2 = set(np.genfromtxt(fp2, dtype=int))
        correct2 = set(np.arange(lang_lengths[lang])).difference(errors2)

        # Top left: Both correct
        table[0,0] += len(correct1.intersection(correct2))
        # Top right: trial 2 incorrect, trial 1 correct
        table[0,1] += len(correct1.intersection(errors2))
        # Bottom left: trial 2 correct, trial 1 incorrect
        table[1,0] += len(errors1.intersection(correct2))
        # Bottom right: Both incorrect
        table[1,1] += len(errors1.intersection(errors2))
    p_val_specificity = mcnemar(table).pvalue
    print('SPECIFICITY')
    print(f'{p_val_specificity:.1e}')
    print(table)
    print()

    print('LINEAR COMBINATION')
    print(f'{p_val_specificity*0.25 + p_val_sensitivity*0.75:.1e}')


if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    main(sys.argv[1], sys.argv[2])