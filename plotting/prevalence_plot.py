import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from pdb import set_trace

def main(file_list):
    percents_dict = {'rm1':0.1162, 'rm2':0.1248, 'rm3':0.1496, 'rm4':0.0880, 'apartment':0.2476, 
                'hotel':0.2347, 'office':0.2502, 'ha':0.0095, 'yo':0.0116, 'bas':0.0175, 'hy-AM':0.0374, 'ckb':0.0460, 
                'cv':0.0113, 'kmr':0.0341, 'ky':0.0141, 'sr':0.0134, 'tt':0.0433}
    n_samples, n_target, n_adapt, prevalences, names = [], [], [], [], []
    file_list.sort()
    for ff in file_list:
        if 'scores.csv' not in ff:
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        run_name = ff.split('/')[-2].split('_')[0]
        if 'n_al' in sheet.keys() and np.array(sheet['n_al'])[-1]>0:
            n_adapt.append(np.sum(sheet['n_al']))
            n_target.append(int(np.sum(sheet['p_target']*sheet['n_al'])))
        else:
            n_adapt.append(np.array(sheet['n_samples'])[-1])
            n_target.append(n_adapt[-1]*np.array(sheet['p_target'])[-1])
        n_samples.append(np.sum(sheet['ps'])+np.sum(sheet['ns']))
        if run_name not in names:
            names.append(run_name)
            prevalences.append(percents_dict[run_name])
    p_target = np.array(n_target)/np.array(n_adapt)
    prevalences = np.array(prevalences)
    orig_adapt = np.array(['Test']*len(prevalences) + ['OAL']*len(prevalences) + ['OAL-CF']*len(prevalences))
    plt.plot([-0.5, len(prevalences)-0.5], [0.5, 0.5], 'k--')
    sns.barplot(x=np.array(names*3), y=np.hstack([prevalences, p_target]), hue=orig_adapt)
    plt.ylabel('% target prevalence')
    fig = plt.gcf()
    fig.set_size_inches(8.5, 3)
    plt.savefig('plotting/plot_out/prevalence.png', dpi=400)


if __name__=='__main__':
    if len(sys.argv) < 3:
        exit()
    main(sys.argv[1:])