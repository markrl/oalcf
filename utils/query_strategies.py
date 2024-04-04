import torch
import torch.nn.functional as F

import pickle
import numpy as np
from skactiveml.pool import RandomSampling, UncertaintySampling, EpistemicUncertaintySampling
from skactiveml.pool import ProbabilisticAL, CostEmbeddingAL
from skactiveml.utils import call_func, MISSING_LABEL

from utils.utils import DcfLoss

from pdb import set_trace

class StrategyManager:
    def __init__(self, params):
        self.combo = params.combo
        self.p_target = params.adapt_distr
        self.est_class = 'both'
        self.min_samples = 0
        if params.combo=='clf':
            with open('clf.p', 'rb') as f:
                self.clf = pickle.load(f)
        elif params.combo=='plateau':
            self.thresh = params.thresh
        
        if 'egl' in params.al_methods.split(','):
            if params.class_loss=='xent':
                self.criterion = torch.nn.NLLLoss(weight=torch.tensor([1, params.target_weight]))
            else:
                self.criterion = DcfLoss()
        
        if 'alce' in params.al_methods.split(','):
            self.cost_obj = CostEmbeddingAL(classes=[0,1])

    def select_queries(self, data_module, method_list, module, n_queries):
        self.logits = self.extract_logits(data_module, module)
        idxs_dict = {}
        rank_dict = {}
        metrics_dict = {}
        if 'rand' in method_list:
            idxs_dict['rand'] = torch.randperm(data_module.unlabeled_len())[:n_queries]
        if 'smax' in method_list:
            smax_out = torch.max(F.softmax(self.logits, dim=-1), dim=-1)[0]
            sorted_idxs = torch.sort(smax_out)[1]
            metrics_dict['smax'] = torch.mean(smax_out)
            if self.combo:
                rank_dict['smax'] = sorted_idxs
            else:
                idxs_dict['smax'] = self.get_top_n(sorted_idxs, n_queries, data_module)
        if 'necs' in method_list:
            neg_energy = -torch.logsumexp(self.logits, dim=-1)
            sorted_idxs = torch.sort(neg_energy, descending=True)[1]
            metrics_dict['necs'] = torch.mean(neg_energy)
            if self.combo:
                rank_dict['necs'] = sorted_idxs
            else:
                idxs_dict['necs'] = self.get_top_n(sorted_idxs, n_queries, data_module)
        if 'ent' in method_list:
            smax_out = F.softmax(self.logits, dim=-1)
            entropy = -torch.sum(smax_out*torch.log(smax_out), dim=-1)
            sorted_idxs = torch.sort(entropy, descending=True)[1]
            metrics_dict['ent'] = torch.mean(entropy)
            if self.combo:
                rank_dict['ent'] = sorted_idxs
            else:
                idxs_dict['ent'] = self.get_top_n(sorted_idxs, n_queries, data_module)
        if 'egl' in method_list:
            smax_out = F.softmax(self.logits, dim=-1)
            grad_lens = extract_grad_lens(data_module, module, self.criterion)
            egl = torch.sum(smax_out*grad_lens, dim=1)
            sorted_idxs = torch.sort(egl, descending=True)[1]
            metrics_dict['egl'] = torch.mean(egl)
            if self.combo:
                rank_dict['egl'] = sorted_idxs
            else:
                idxs_dict['egl'] = self.get_top_n(sorted_idxs, n_queries, data_module)
        if 'pert' in method_list:
            change_percentages = perturbate(data_module, module)
            sorted_idxs = torch.sort(change_percentages, descending=True)[1]
            metrics_dict['pert'] = torch.mean(change_percentages)
            if self.combo:
                rank_dict['pert'] = sorted_idxs
            else:
                idxs_dict['pert'] = self.get_top_n(sorted_idxs, n_queries, data_module)
        if 'cover' in method_list:
            idxs_dict['cover'] = cover_distribution(data_module, n_queries)
            metrics_dict['cover'] = 1
        if 'alce' in method_list:
            y = data_module.get_train_labels()
            y_cand = torch.full(size=(self.logits.shape[0],), fill_value=MISSING_LABEL)
            y = torch.cat([y_cand,y], dim=0)
            X = self.extract_features(data_module)
            idxs_dict['alce'] = self.cost_obj.query(X, y, batch_size=n_queries)
            metrics_dict['alce'] = 1

        if self.combo=='rank':
            all_ranks = [rank_dict[kk] for kk in rank_dict.keys()]
            combo_rank = torch.sum(torch.stack(all_ranks,dim=0),dim=0)
            sorted_idxs = torch.sort(combo_rank)[1]
            idxs_dict['combo'] = self.get_top_n(sorted_idxs, n_queries, data_module)
        elif self.combo=='plateau':
            if metrics_dict['smax'] > self.thresh:
                idxs_dict['combo'] = rank_dict['necs'][:n_queries]
            else:
                idxs_dict['combo'] = rank_dict['smax'][:n_queries]
        elif self.combo=='clf':
            feat = np.vstack([metrics_dict['smax'], metrics_dict['necs'], len(data_module.data_train)+n_queries]).T
            strategy_idx = self.clf.predict(feat)[0]
            if strategy_idx==0:
                idxs_dict['combo'] = rank_dict['smax'][:n_queries]
            else:
                idxs_dict['combo'] = rank_dict['necs'][:n_queries]
        elif self.combo=='rand':
            if torch.randint(2, [1])[0]==0:
                idxs_dict['combo'] = rank_dict['smax'][:n_queries]
            else:
                idxs_dict['combo'] = rank_dict['necs'][:n_queries]
        elif self.combo=='split':
            n_queries_0 = int(n_queries/2)
            n_queries_1 = n_queries-n_queries_0
            for qq in range(n_queries_0):
                rank_dict['necs'] = rank_dict['necs'][rank_dict['necs']!=rank_dict['smax'][qq]]
            idxs_dict['combo'] = torch.cat([rank_dict['smax'][:n_queries_0], 
                                            rank_dict['necs'][:n_queries_1]])
        
        if len(idxs_dict)==0:
            exit()
        return idxs_dict, metrics_dict

    def get_top_n(self, sorted_idxs, n_queries, data_module):
        if self.p_target is None:
            return sorted_idxs[:n_queries]
        else:
            if self.est_class=='both':
                preds = torch.argmax(F.softmax(self.logits, dim=-1), dim=-1)
                t_idxs = torch.where(preds==1)[0]
                n_idxs = torch.where(preds==0)[0]
                M_t = len(t_idxs)
                M_n = len(n_idxs)
                N_t,N_n = data_module.get_class_counts()
                L_t = int(torch.round(self.p_target*(N_t+N_n+n_queries)-N_t))
                upper_bound = torch.min(torch.tensor([n_queries, M_t]))
                L_t = int(torch.clamp(torch.tensor(L_t), 0, upper_bound))
                L_n = n_queries-L_t
                rank_t = torch.tensor([ii for ii in sorted_idxs if ii in t_idxs])
                rank_n = torch.tensor([ii for ii in sorted_idxs if ii in n_idxs])
                return torch.cat([rank_t[:L_t], rank_n[:L_n]], dim=0).long()
            elif self.est_class=='target':
                preds = torch.argmax(F.softmax(self.logits, dim=-1), dim=-1)
                t_idxs = torch.where(preds==1)[0]
                M_t = len(t_idxs)
                N_t,N_n = data_module.get_class_counts()
                L_t = (self.p_target*(N_t+N_n)-N_t)/(1-self.p_target)
                L_t = int(torch.round(L_t))
                upper_bound = torch.min(torch.tensor([n_queries, M_t]))
                lower_bound = torch.min(torch.tensor([self.min_samples, upper_bound]))
                L_t = int(torch.clamp(torch.tensor(L_t), lower_bound, upper_bound))
                rank_t = torch.tensor([ii for ii in sorted_idxs if ii in t_idxs])
                return rank_t[:L_t]
            elif self.est_class=='nontarget':
                preds = torch.argmax(F.softmax(self.logits, dim=-1), dim=-1)
                n_idxs = torch.where(preds==0)[0]
                M_n = len(n_idxs)
                N_t,N_n = data_module.get_class_counts()
                L_n = ((1-self.p_target)*(N_t+N_n)-N_n)/self.p_target
                L_n = int(torch.round(L_n))
                upper_bound = torch.min(torch.tensor([n_queries, M_n]))
                lower_bound = torch.min(torch.tensor([self.min_samples, upper_bound]))
                L_n = int(torch.clamp(torch.tensor(L_n), lower_bound, upper_bound))
                rank_n = torch.tensor([ii for ii in sorted_idxs if ii in n_idxs])
                return rank_n[:L_n]

    def extract_logits(self, data_module, module):
        model = module.model
        model.eval()
        if hasattr(data_module, 'unlabeled_dataloader'):
            loader = data_module.unlabeled_dataloader()
        else:
            loader = data_module.test_dataloader()
        with torch.no_grad():
            if torch.cuda.is_available():
                model = model.cuda()
                logits = [model(batch[0].cuda())[-1] for batch in loader]
            else:
                logits = [model(batch[0])[-1] for batch in loader]
        logits = torch.cat(logits, dim=0).cpu()
        return logits

    def extract_features(self, data_module):
        test_loader = data_module.test_dataloader()
        train_loader = data_module.train_dataloader()
        test_feats = torch.cat([batch[0] for batch in test_loader], dim=0)
        train_feats = torch.cat([batch[0] for batch in train_loader], dim=0)
        feats = torch.cat([test_feats, train_feats], dim=0)
        return feats

def extract_grad_lens(data_module, module, criterion):
    model = module.model
    if hasattr(data_module, 'unlabeled_dataloader'):
        loader = data_module.unlabeled_dataloader()
    else:
        loader = data_module.test_dataloader()
    all_grad_lens = []
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()
        for batch in loader:
            samples = batch[0].cuda()
            samples.requires_grad = True
            for x in samples:
                for param in model.parameters():
                    param.grad = None
                    param.requires_grad = True
                out = model(torch.stack([x,x], dim=0))[1]
                loss = criterion(out, torch.zeros(out.shape[0]).cuda().long())
                grad = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
                grad = [gg.reshape(-1) for gg in grad if gg is not None]
                grad_len = [torch.linalg.vector_norm(torch.cat(grad))]

                for param in model.parameters():
                    param.grad = None
                    param.requires_grad = True
                out = model(torch.stack([x,x], dim=0))[1]
                loss = criterion(out, torch.ones(out.shape[0]).cuda().long())
                grad = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
                grad = [gg.reshape(-1) for gg in grad if gg is not None]
                grad_len.append(torch.linalg.vector_norm(torch.cat(grad)))
                all_grad_lens.append(torch.stack(grad_len))
                
    else:
        for batch in loader:
            samples = batch[0]
            samples.requires_grad = True
            grad = []
            for x in samples:
                for param in model.parameters():
                    param.requires_grad = True
                out = model(torch.stack([x,x], dim=0))[1]
                loss = criterion(out, torch.zeros(out.shape[0]).long())
                grad = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
                grad = [gg.reshape(-1) for gg in grad if gg is not None]
                grad_len = [torch.linalg.vector_norm(torch.cat(grad))]

                for param in model.parameters():
                    param.grad = None
                    param.requires_grad = True
                out = model(torch.stack([x,x], dim=0))[1]
                loss = criterion(out, torch.ones(out.shape[0]).long())
                grad = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
                grad = [gg.reshape(-1) for gg in grad if gg is not None]
                grad_len.append(torch.linalg.vector_norm(torch.cat(grad)))
                all_grad_lens.append(torch.cat(grad_len))
    return torch.stack(all_grad_lens, dim=0).cpu()

def get_dists(samples, keep_same_idx=True, until_idx=None):
    all_dists = []
    for ii,sample in enumerate(samples):
        if ii == until_idx:
            break
        diff = samples-sample
        dists = torch.norm(diff, dim=1)
        if not keep_same_idx:
            dists = torch.cat((dists[:ii], dists[ii+1:]))
        all_dists.append(dists)
    return torch.stack(all_dists, dim=0)

def perturbate(data_module, module):
    # Prep model and dataloader
    model = module.model
    if torch.cuda.is_available():
        model = model.cuda()
    if hasattr(data_module, 'unlabeled_dataloader'):
        loader = data_module.unlabeled_dataloader()
    else:
        loader = data_module.test_dataloader()

    # Set epsilon
    # samples = []
    # for batch in loader:
    #     samples.append(batch[0])
    # samples = torch.cat(samples, dim=0)
    # dists = get_dists(samples, False)
    # eps = torch.min(dists)/2
    eps = 0.001
    
    change_percentages = []
    for batch in loader:
        samples = batch[0]
        if torch.cuda.is_available():
            samples = samples.cuda()
        for x in samples:
            preds = []
            new_batch = torch.zeros(samples.shape)
            all_perts = eps*torch.cat([torch.zeros(1,samples.shape[1]), 
                                    torch.eye(samples.shape[1]), 
                                    -torch.eye(samples.shape[1])], dim=0)
            if torch.cuda.is_available():
                new_batch = new_batch.cuda()
                all_perts = all_perts.cuda()
            for ii,pert in enumerate(all_perts):
                jj = ii%new_batch.shape[0]
                new_batch[jj] = x + pert
                if jj==new_batch.shape[0]-1 or ii==all_perts.shape[0]-1:
                    with torch.no_grad():
                        scores = model(new_batch)[1]
                    preds.append(torch.argmax(scores, dim=-1).cpu().float())
            preds = torch.cat(preds)[:all_perts.shape[0]]
            change_percentages.append(torch.mean(1.0*(preds!=preds[0])))
    return torch.FloatTensor(change_percentages)

def cover_distribution(data_module, n_queries):
    # Get distances
    if hasattr(data_module, 'unlabeled_dataloader'):
        loader = data_module.unlabeled_dataloader()
    else:
        loader = data_module.test_dataloader()
    new_samples = []
    for batch in loader:
        new_samples.append(batch[0])
    new_samples = torch.cat(new_samples, dim=0)

    loader = data_module.train_dataloader()
    labeled_samples = []
    for batch in loader:
        labeled_samples.append(batch[0])
    labeled_samples = torch.cat(labeled_samples, dim=0)

    samples = torch.cat([new_samples, labeled_samples], dim=0)
    D = get_dists(samples, True, len(new_samples))
    d_orig, _ = torch.min(D[:,len(new_samples):], dim=-1)
    d = D[:,:len(new_samples)]

    idxs = []
    for ii in range(n_queries):
        min_t = torch.sum(d_orig)
        for ii,d_i in enumerate(d):
            d_new_i, _ = torch.min(torch.stack([d_orig,d_i], dim=0), dim=0)
            t_i = torch.sum(d_new_i)
            if t_i < min_t:
                min_t = t_i
                min_d_new = d_new_i
                min_idx = ii
        idxs.append(min_idx)
        d_orig = min_d_new
    return torch.LongTensor(idxs)
