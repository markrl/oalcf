import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(csv_paths, metric=None, outpath=None):
    if metric is None:
        metric = 'dcf'
    if outpath is None:
        outpath = '/home/marklind/research/AL/vtd_iml/plotting/plot_out/comp_env.png'
    csv_files = csv_paths.split(',')
    for ff in csv_files:
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
        else:
            scores = np.array(sheet[metric])
        scores[np.isnan(scores)] = 0
        print(f'{scores[-1]:.4f}')
        plt.plot(scores, label=ff.split('/')[-2])
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