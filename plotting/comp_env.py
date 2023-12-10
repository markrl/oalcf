import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(csv_paths, metric=None, label_names=None, outpath=None):
    if metric is None:
        metric = 'dcf'
    if outpath is None:
        outpath = '/home/marklind/research/AL/vtd_iml/plotting/plot_out/comp_env.png'
    csv_files = csv_paths.split(',')
    for ii,ff in enumerate(csv_files):
        if os.path.basename(ff) != 'scores.csv':
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        if metric=='dcf':
            scores = np.array(sheet['cum_dcf'])
        elif metric=='fnr':
            fns = np.cumsum(sheet['fns'])
            ps = np.cumsum(sheet['ps'])
            scores = fns/ps
        elif metric=='fpr':
            fns = np.cumsum(sheet['fps'])
            ps = np.cumsum(sheet['ns'])
            scores = fps/ns
        elif metric=='new_samples':
            scores = np.array(sheet['n_samples'])
            scores = np.diff(scores, prepend=8)
        else:
            scores = np.array(sheet[metric])
        scores[np.isnan(scores)] = 0
        print(f'{scores[-1]:.4f}')
        if label_names is None:
            plt.plot(scores, label=ff.split('/')[-2])
        else:
            plt.plot(scores, label=label_names.split(',')[ii])
    plt.xlabel('Session')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.savefig(outpath)
    plt.clf()


if __name__=='__main__':
    if len(sys.argv)<2:
        exit()
    if len(sys.argv)==2:
        main(sys.argv[1])
    elif len(sys.argv)==3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])