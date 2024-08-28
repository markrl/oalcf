import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace

def main(dir_code, metric='dcf', pre_post_diff='post', corpus_task_paradigm='corpus'):
    outpath = 'plotting/plot_out/instant.png'
    corpus_dict = {'sri': ['rm1', 'rm2', 'rm3', 'rm4'], 'lb': ['apartment', 'hotel', 'office'],
                    'ac': ['yo', 'ha', 'bas'], 'cr': ['ckb', 'cv', 'kmr', 'tt', 'hy-AM', 'sr', 'ky']}
    task_dict = {'vtd': ['rm1', 'rm2', 'rm3', 'rm4', 'apartment', 'hotel', 'office'],
                    'slv': ['yo', 'ha', 'bas', 'ckb', 'cv', 'kmr', 'tt', 'hy-AM', 'sr', 'ky']}
    paradigm_dict = {'oal': ['oal'], 'oal-cf': ['oalcf'], 'pml': ['pml']}
    run_dict = {'random':'rand', 'alce':'alceal', 'necs':'necsal', 'smax':'smaxal', 'egl':'eglal', 
                'cdba':'cdba', 'ddba':'ddba', 'cdba (oal)':'cdbaoal', 'uniform (oal)': 'uniformoal'}
    ps, ns, fps, fns = [], [], [], []
    pre_ps, pre_ns, pre_fps, pre_fns = [], [], [], []
    pre_n_adapt, n_adapt, n_al = [], [], []
    al_metric, p_target = [], []
    corpora, tasks, paradigms, runs = [], [], [], []
    dir_code = dir_code.replace('%', '*')
    file_list = []
    for dc in dir_code.split(','):
        file_list += glob.glob(os.path.join(dc, '*', 'scores.csv'))
    # file_list.sort()
    for ff in file_list:
        sheet = pd.read_csv(ff)
        env_name = ff.split('/')[-2].split('_')[0]
        paradigm_name = ff.split('/')[-3].split('_')[1]
        run_name = ff.split('/')[-3].split('_')[2]
        n_bootstrap = sheet['n_samples'][0]-sheet['n_al'][0]
        ps.append(np.array(sheet['ps']))
        ns.append(np.array(sheet['ns']))
        fps.append(np.array(sheet['fps']))
        fns.append(np.array(sheet['fns']))
        n_adapt.append(n_bootstrap+np.array(sheet['n_al']))
        pre_ps.append(np.array(sheet['pre_ps']))
        pre_ns.append(np.array(sheet['pre_ns']))
        pre_fps.append(np.array(sheet['pre_fps']))
        pre_fns.append(np.array(sheet['pre_fns']))
        pre_n_adapt.append(n_bootstrap+np.array(sheet['n_al'])-sheet['n_al'][0])
        al_metric.append(np.array(sheet['metric']))
        n_al.append(np.array(sheet['n_al']))
        p_target.append(np.array(sheet['p_target']))
        for kk in corpus_dict:
            if env_name in corpus_dict[kk]:
                corpora.append(kk)
        for kk in task_dict:
            if env_name in task_dict[kk]:
                tasks.append(kk)
        for kk in paradigm_dict:
            if paradigm_name in paradigm_dict[kk]:
                paradigms.append(kk)
        for kk in run_dict:
            if run_name==run_dict[kk]:
                runs.append(kk)
        if len(runs) < len(paradigms):
            run_dict[run_name] = [run_name]
            runs.append(run_name)

    if pre_post_diff=='post':
        fprs = [np.nan_to_num(fp/n) for fp,n in zip(fps,pre_ns)]
        fnrs = [np.nan_to_num(fn/p) for fn,p in zip(fns,pre_ps)]
        dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(fnrs,fprs)]
        imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(fps,n_adapt,pre_ns,pre_ps,fns)]
    elif pre_post_diff=='pre':
        fprs = [np.nan_to_num(fp/n) for fp,n in zip(pre_fps,pre_ns)]
        fnrs = [np.nan_to_num(fn/p) for fn,p in zip(pre_fns,pre_ps)]
        dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(fnrs,fprs)]
        imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(pre_fps,pre_n_adapt,pre_ns,pre_ps,pre_fns)]
    else:
        post_fprs = [np.nan_to_num(fp/n) for fp,n in zip(fps,pre_ns)]
        post_fnrs = [np.nan_to_num(fn/p) for fn,p in zip(fns,pre_ps)]
        post_dcfs = [0.75*fnr+0.25*fpr for fnr,fpr in zip(post_fnrs,post_fprs)]
        post_imlms = [(fp+na)/(n+p)+np.nan_to_num(fn/p) for fp,na,n,p,fn in zip(fps,n_adapt,pre_ns,pre_ps,fns)]
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
    elif metric=='plateau':
        scores = al_metric
    elif metric=='n_al':
        scores = n_al
    elif metric=='p_target':
        scores = p_target

    for ss,ff in zip(scores,file_list):
        dim1 = ss[:-1]
        dim2 = ss[1:]
        cc = np.corrcoef(dim1,dim2)[0,1]
        print(f'{ff}:{cc:.4f}')

    if corpus_task_paradigm=='corpus':
        decider = corpora
        decider_dict = corpus_dict
    elif corpus_task_paradigm=='task':
        decider = tasks
        decider_dict = task_dict
    elif corpus_task_paradigm=='paradigm':
        decider = paradigms
        decider_dict = paradigm_dict
    elif corpus_task_paradigm=='run':
        decider = runs
        decider_dict = run_dict
    
    scores_dict = {}
    for kk in decider_dict:
        corpus_scores = [ss for cc,ss in zip(decider,scores) if cc==kk]
        if len(corpus_scores) > 0:
            all_ccs = []
            for ss in corpus_scores:
                dim1 = ss[:-1]
                dim2 = ss[1:]
                cc = np.corrcoef(dim1,dim2)[1,0]
                all_ccs.append(cc)
            print(f'{kk},mean:{np.mean(all_ccs):.4f}')
            min_len = np.min([len(ss) for ss in corpus_scores])
            corpus_scores = [ss[:min_len] for ss in corpus_scores]
            corpus_scores = np.mean(corpus_scores, axis=0)
            dim1 = corpus_scores[:-1]
            dim2 = corpus_scores[1:]
            cc = np.corrcoef(dim1,dim2)[1,0]
            print(f'mean,{kk}:{cc:.4f}')
            if metric=='plateau':
                corpus_scores = corpus_scores-corpus_scores.min()
                corpus_scores = corpus_scores/corpus_scores.max()
                if corpus_scores[0] > corpus_scores[-1]:
                    corpus_scores = 1 - corpus_scores
            if kk=='uniform':
                label = 'Uniform'
            elif kk=='linear12':
                label = 'Linear'
            elif kk=='exponent':
                label = 'Exponential'
            elif kk=='frontheavy':
                label = 'Frontheavy'
            else:
                label = kk.upper()
            plt.plot(corpus_scores, label=label)# + ' (OAL)')
    plt.xlabel('Session')
    if metric=='plateau':
        plt.ylabel('Scaled AL Strategy Metric')
    elif metric=='n_al':
        plt.ylabel('# AL queries')
    elif metric=='p_target':
        plt.ylabel('% target class prevalence')
    else:
        plt.ylabel(metric.upper())
    plt.xlim([0, min_len-1])
    # plt.ylim([0, 0.45])
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
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])