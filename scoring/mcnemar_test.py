import os
import sys
import glob
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

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
                    'yue':90720,}

    fns1, fps1 = []
    for dd in run1.split(','):
        fns1 += glob.glob(os.path.join(dd, '*', 'fn_list.txt'))
        fps1 += glob.glob(os.path.join(dd, '*', 'fp_list.txt'))
    fns1.sort()
    fps1.sort()

    fns2, fps2 = []
    for dd in run2.split(','):
        fns2 += glob.glob(os.path.join(dd, '*', 'fn_list.txt'))
        fps2 += glob.glob(os.path.join(dd, '*', 'fp_list.txt'))
    fns2.sort()
    fps2.sort()

if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    main(sys.argv[1], sys.argv[2])