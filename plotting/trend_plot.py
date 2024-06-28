import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace

def main(dir_code, metric='dcf', pre_post_diff='post'):
    outpath = 'plotting/plot_out/trends.png'
    corpus_dict = {'sri': ['rm1', 'rm2', 'rm3', 'rm4'], 'lb': ['apartment', 'hotel', 'office'],
                    'ac': ['yo', 'ha', 'bas'], 'cr': ['ckb', 'cv', 'kmr', 'tt', 'hy-AM', 'sr', 'ky']}
    ps, ns, fps, fns = [], [], [], []
    pre_ps, pre_ns, pre_fps, pre_fns = [], [], [], []
    pre_n_adapt, n_adapt, corpora = [], [], []
    if dir_code[-1] == '_':
        dir_code = dir_code[:-1]
    dir_components = dir_code.split('_')
    dirs = ['_'.join(dir_components + ['vtd']), '_'.join(dir_components + ['lid'])]
    file_list = glob.glob(os.path.join('output', dirs[0], '*')) + glob.glob(os.path.join('output', dirs[1], '*'))
    for ff in file_list:
        if 'scores.csv' not in ff:
            ff = os.path.join(ff, 'scores.csv')
        sheet = pd.read_csv(ff)
        run_name = ff.split('/')[-2].split('_')[0]
        n_bootstrap = sheet['n_samples'][0]-sheet['n_al'][0]
        ps.append(np.cumsum(np.array(sheet['ps'])))
        ns.append(np.cumsum(np.array(sheet['ns'])))
        fps.append(np.cumsum(np.array(sheet['fps'])))
        fns.append(np.cumsum(np.array(sheet['fns'])))
        n_adapt.append(n_bootstrap+np.cumsum(np.array(sheet['n_al'])))
        pre_ps.append(np.cumsum(np.array(sheet['pre_ps'])))
        pre_ns.append(np.cumsum(np.array(sheet['pre_ns'])))
        pre_fps.append(np.cumsum(np.array(sheet['pre_fps'])))
        pre_fns.append(np.cumsum(np.array(sheet['pre_fns'])))
        pre_n_adapt.append(n_bootstrap+np.cumsum(np.array(sheet['n_al']))-sheet['n_al'][0])
        for kk in corpus_dict:
            if run_name in corpus_dict[kk]:
                corpora.append(kk)

    if pre_post_diff=='post':
        fprs = [np.nan_to_num(fp/n) for fp,n in zip(fps,ns)]
        fnrs = [np.nan_to_num(fn/p) for fn,p in zip(fns,ps)]
        dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(fnrs,fprs)]
        imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(fps,n_adapt,ns,ps,fns)]
    elif pre_post_diff=='pre':
        fprs = [np.nan_to_num(fp/n) for fp,n in zip(pre_fps,pre_ns)]
        fnrs = [np.nan_to_num(fn/p) for fn,p in zip(pre_fns,pre_ps)]
        dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(fnrs,fprs)]
        imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(pre_fps,pre_n_adapt,pre_ns,pre_ps,pre_fns)]
    else:
        post_fprs = [np.nan_to_num(fp/n) for fp,n in zip(fps,ns)]
        post_fnrs = [np.nan_to_num(fn/p) for fn,p in zip(fns,ps)]
        post_dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(post_fnrs,post_fprs)]
        post_imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(fps,n_adapt,ns,ps,fns)]
        pre_fprs = [np.nan_to_num(fp/n) for fp,n in zip(pre_fps,pre_ns)]
        pre_fnrs = [np.nan_to_num(fn/p) for fn,p in zip(pre_fns,pre_ps)]
        pre_dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(pre_fnrs,pre_fprs)]
        pre_imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(pre_fps,pre_n_adapt,pre_ns,pre_ps,pre_fns)]
        fprs = [pre-post for pre,post in zip(pre_fprs,post_fprs)]
        fnrs = [pre-post for pre,post in zip(pre_fnrs,post_fnrs)]
        dcfs = [pre-post for pre,post in zip(pre_dcfs,post_dcfs)]
        imlms = [pre-post for pre,post in zip(pre_imlms,post_imlms)]
    
    if metric=='dcf':
        scores = dcfs
    elif metric=='fnr':
        scores = fnrs
    elif metric=='fpr':
        scores = fpr
    elif metric=='imlm':
        scores = imlms
    
    scores_dict = {}
    for kk in corpus_dict:
        corpus_scores = [ss for cc,ss in zip(corpora,scores) if cc==kk]
        min_len = np.min([len(ss) for ss in corpus_scores])
        corpus_scores = [ss[:min_len] for ss in corpus_scores]
        plt.plot(np.mean(corpus_scores, axis=0), label=kk.upper())
    plt.xlabel('Session')
    plt.ylabel(metric.upper())
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(8.5, 3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=400)

if __name__=='__main__':
    if len(sys.argv) < 2:
        exit()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])