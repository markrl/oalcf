import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def main(budget_paths, label_names=None, accumulate=False):
    budget_paths = budget_paths.strip().split(',')
    if label_names is not None:
        label_names = label_names.strip().split(',')
    else:
        label_names = [nn+1 for nn in range(len(budget_paths))]
    maxs = []
    for ii,ff in enumerate(budget_paths):
        if 'budget.txt' not in ff:
            ff = os.path.join(ff, 'budget.txt')
        trace = np.genfromtxt(ff)
        if accumulate:
            trace = np.cumsum(trace)
        plt.plot(trace, label=label_names[ii])
        maxs.append(trace.max())
    plt.xlabel('Session')
    if accumulate:
        plt.ylabel('Accumulated samples')
    else:
        plt.ylabel('# samples')
    plt.legend()
    plt.xlim([-1,len(trace)])
    plt.ylim([0,np.max(maxs)*1.05])
    fig = plt.gcf()
    fig.set_size_inches(8.5, 3)
    plt.tight_layout()
    plt.savefig('/home/marklind/research/AL/vtd_iml/plotting/plot_out/budget.png')
    plt.clf()

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    elif len(sys.argv)==2:
        main(sys.argv[1])
    elif len(sys.argv)==3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], True)