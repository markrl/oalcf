import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus

from pdb import set_trace

def logits2neg_energy(logits, T=1.0):
    neg_energy_score = -T*torch.logsumexp(logits/T, dim=1)
    return neg_energy_score


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0, weight=1.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        # self.dist_func = euclidean_distance
        self.dist_func = cosine_distance
        self.reduction = reduction

    def forward(self, x0, x1, y0, y1):
        y = 1*(y0==y1)
        weight = (self.weight-1)*torch.logical_and(y0==1, y1==1) + 1
        dist = self.dist_func(x0, x1)
        dist_sq = dist**2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        if self.reduction=='mean':
            loss = torch.sum(loss * weight) / 2.0 / x0.size()[0]
            return loss
        elif self.reduction=='sum':
            loss = torch.sum(loss)
        else:
            return loss


def euclidean_distance(x0, x1):
    diff = x0 - x1
    return torch.sqrt(torch.sum(torch.pow(diff, 2), 1))

def cosine_distance(x0, x1):
    return 1 - F.cosine_similarity(x0, x1, dim=1)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        centers_batch = self.centers[labels]
        dist = torch.pow(x - centers_batch, 2).sum(dim=1).sqrt()
        loss = dist.mean() / 2.0

        # Update centers
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = labels == i
                if mask.sum() > 0:
                    center_delta = (x[mask] - self.centers[i]).mean(dim=0)
                    self.centers[i] = self.centers[i] + self.alpha * center_delta

        return loss


class DcfLoss(torch.nn.Module):
    def __init__(self, fnr_weight=0.75, smax_weight=0, learn_mult=False, learn_error_weight=False):
        super(DcfLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(fnr_weight).float(), requires_grad=learn_error_weight)
        self.beta = smax_weight
        self.smax = torch.nn.Softmax(dim=-1)
        self.mult = nn.Parameter(torch.tensor(1.).float(), requires_grad=learn_mult)

    def forward(self, x, y):
        '''
        x: Log of class posterior probabilities (on the interval [-inf,inf])
        y: Labels (0 or 1)
        '''
        self.alpha.data = torch.clamp(self.alpha.data, min=0.05, max=0.95)
        x = torch.exp(x)
        if self.beta>0:
            x = self.smax(self.beta*self.mult*x)
        expected_fnr = torch.dot(x[:,0],y*1.0)/torch.sum(y) if torch.sum(y)>0 else 0
        expected_fpr = torch.dot(x[:,1],(1.0-y))/torch.sum(1-y) if torch.sum(1-y)>0 else 0
        dcf = self.alpha*expected_fnr + (1-self.alpha)*expected_fpr
        return dcf


class ImlmLoss(torch.nn.Module):
    def __init__(self, smax_weight=0, learn_mult=False):
        super(ImlmLoss, self).__init__()
        self.beta = smax_weight
        self.smax = torch.nn.Softmax(dim=-1)
        self.mult = nn.Parameter(torch.tensor(1.).float(), requires_grad=learn_mult)

    def forward(self, x, y):
        '''
        x: Log of class posterior probabilities (on the interval [-inf,inf])
        y: Labels (0 or 1)
        '''
        x = torch.exp(x)
        if self.beta>0:
            x = self.smax(self.beta*self.mult*x)
        expected_neg_cost = torch.dot(x[:,0],y*1.0)/torch.sum(y) if torch.sum(y)>0 else 0
        expected_pos_cost = torch.dot(x[:,1],(1.0-y))/len(y)
        imlm = expected_pos_cost + expected_neg_cost
        return imlm


class aDcfLoss(torch.nn.Module):
    def __init__(self, fnr_weight=0.75, temp=40):
        super(aDcfLoss, self).__init__()
        self.temp = temp
        self.alpha = fnr_weight
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        '''
        x: Log of class posterior probabilities (on the interval [-inf,inf])
        y: Labels (0 or 1)
        '''
        x = torch.exp(x)
        targ_mask = y==1
        non_mask = y==0
        p_fn = torch.sum(self.sig(self.temp*(0.5-x[targ_mask,1])))/len(targ_mask) if len(targ_mask)>0 else 0
        p_fp = torch.sum(self.sig(self.temp*(x[non_mask,1]-0.5)))/len(non_mask) if len(non_mask)>0 else 0
        dcf = self.alpha*p_fn + (1-self.alpha)*p_fp
        return dcf


class FocalLoss(nn.Module):
    # From https://github.com/clcarwin/focal_loss_pytorch
    def __init__(self, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt.data)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss
        

class LearnableNLLLoss(nn.Module):
    def __init__(self,
            weight = None,
            reduction = "mean",):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.learned_mult = nn.Parameter(torch.tensor(1.).float(), requires_grad=True)

    def forward(self, inp, target):
        weights_mult = torch.ones(target.shape)
        weights_mult[target==1] *= self.learned_mult*self.weight/(self.learned_mult*self.weight+1)
        weights_mult[target==0] *= 1/(self.learned_mult*self.weight+1)
        losses = -weights_mult*(target*inp[:,1] + (1-target)*inp[:,0])
        # weights_frac = torch.ones(target.shape)
        # weights_frac[target==1] /= torch.sum(weights_mult[target==1])
        # weights_frac[target==0] /= torch.sum(weights_mult[target==0])
        # if self.reduction=='mean':
        #     return torch.mean(weights_frac*losses)
        if self.reduction=='mean':
            return torch.mean(losses)
        else:
            return losses
        

def choose_pair(embeds, labels, pair_type):
    target_idxs = torch.where(labels==1)[0]
    nontarget_idxs = torch.where(labels==0)[0]
    if 'rand' in pair_type:
        return torch.randint(low=0,high=len(embeds),size=(len(embeds),))
    elif 'semi' in pair_type:
        dists = []
        for embed in embeds:
            dists.append(1-torch.cosine_similarity(embed, embeds))
        dists = torch.stack(dists, dim=0)
        same_idxs = torch.randint(high=2,size=(len(embeds),))
        out_idxs = torch.zeros(len(embeds), dtype=torch.long)
        if len(target_idxs)>0:
            if torch.sum(torch.logical_and(same_idxs==1, labels==1)) > 0:
                out_idxs[torch.logical_and(same_idxs==1, labels==1)] = target_idxs[torch.argmax(dists[torch.logical_and(same_idxs==1, labels==1)][:,target_idxs], dim=-1)]
        if len(nontarget_idxs)>0:
            if torch.sum(torch.logical_and(same_idxs==1, labels==0)) > 0:
                out_idxs[torch.logical_and(same_idxs==1, labels==0)] = nontarget_idxs[torch.argmax(dists[torch.logical_and(same_idxs==1, labels==0)][:,nontarget_idxs], dim=-1)]
        if len(target_idxs)>0 and len(nontarget_idxs)>0:
            if torch.sum(torch.logical_and(same_idxs==0, labels==0)) > 0:
                out_idxs[torch.logical_and(same_idxs==0, labels==0)] = target_idxs[torch.argmin(dists[torch.logical_and(same_idxs==0, labels==0)][:,target_idxs], dim=-1)]
            if torch.sum(torch.logical_and(same_idxs==0, labels==1)) > 0:
                out_idxs[torch.logical_and(same_idxs==0, labels==1)] = nontarget_idxs[torch.argmin(dists[torch.logical_and(same_idxs==0, labels==1)][:,nontarget_idxs], dim=-1)]
        return out_idxs
    elif 'hard' in pair_type:
        dists = []
        for embed in embeds:
            dists.append(1-torch.cosine_similarity(embed, embeds))
        dists = torch.stack(dists, dim=0)
        same_idxs = torch.randint(high=2,size=(len(embeds),))
        out_idxs = torch.zeros(len(embeds), dtype=torch.long)
        if len(target_idxs)>0:
            if torch.sum(torch.logical_and(same_idxs==1, labels==1)) > 0:
                out_idxs[torch.logical_and(same_idxs==1, labels==1)] = target_idxs[torch.argmax(dists[torch.logical_and(same_idxs==1, labels==1)][:,target_idxs], dim=-1)]
        if len(nontarget_idxs)>0:
            if torch.sum(torch.logical_and(same_idxs==1, labels==0)) > 0:
                out_idxs[torch.logical_and(same_idxs==1, labels==0)] = nontarget_idxs[torch.argmax(dists[torch.logical_and(same_idxs==1, labels==0)][:,nontarget_idxs], dim=-1)]
        if len(target_idxs)>0 and len(nontarget_idxs)>0:
            if torch.sum(torch.logical_and(same_idxs==0, labels==0)) > 0:
                out_idxs[torch.logical_and(same_idxs==0, labels==0)] = target_idxs[torch.argmin(dists[torch.logical_and(same_idxs==0, labels==0)][:,target_idxs], dim=-1)]
            if torch.sum(torch.logical_and(same_idxs==0, labels==1)) > 0:
                out_idxs[torch.logical_and(same_idxs==0, labels==1)] = nontarget_idxs[torch.argmin(dists[torch.logical_and(same_idxs==0, labels==1)][:,nontarget_idxs], dim=-1)]
        return out_idxs


def choose_pos_neg(embeds, labels, pair_type):
    target_idxs = torch.where(labels==1)[0]
    nontarget_idxs = torch.where(labels==0)[0]
    if 'rand' in pair_type:
        pos_idxs = torch.zeros(len(embeds), dtype=torch.long)
        neg_idxs = torch.zeros(len(embeds), dtype=torch.long)
        pos_idxs[target_idxs] = target_idxs[torch.randint(low=0,high=len(target_idxs),size=(len(target_idxs),))]
        neg_idxs[target_idxs] = nontarget_idxs[torch.randint(low=0,high=len(nontarget_idxs),size=(len(target_idxs),))]
        pos_idxs[nontarget_idxs] = nontarget_idxs[torch.randint(low=0,high=len(nontarget_idxs),size=(len(nontarget_idxs),))]
        neg_idxs[nontarget_idxs] = target_idxs[torch.randint(low=0,high=len(target_idxs),size=(len(nontarget_idxs),))]
    elif 'semi' in pair_type:
        dists = []
        for embed in embeds:
            dists.append(1-torch.cosine_similarity(embed, embeds))
        dists = torch.stack(dists, dim=0)
        pos_idxs = torch.zeros(len(embeds), dtype=torch.long)
        neg_idxs = torch.zeros(len(embeds), dtype=torch.long)
        pos_idxs[target_idxs] = target_idxs[torch.argmax(dists[target_idxs][:,target_idxs], dim=-1)]
        pos_idxs[nontarget_idxs] = nontarget_idxs[torch.argmax(dists[nontarget_idxs][:,nontarget_idxs], dim=-1)]
        pos_dists = dists[pos_idxs,pos_idxs]
        neg_idxs[target_idxs] = nontarget_idxs[torch.argmin(dists[target_idxs][:,nontarget_idxs], dim=-1)]
        neg_idxs[nontarget_idxs] = target_idxs[torch.argmin(dists[nontarget_idxs][:,target_idxs], dim=-1)]
    elif 'hard' in pair_type:
        dists = []
        for embed in embeds:
            dists.append(1-torch.cosine_similarity(embed, embeds))
        dists = torch.stack(dists, dim=0)
        pos_idxs = torch.zeros(len(embeds), dtype=torch.long)
        neg_idxs = torch.zeros(len(embeds), dtype=torch.long)
        pos_idxs[target_idxs] = target_idxs[torch.argmax(dists[target_idxs][:,target_idxs], dim=-1)]
        pos_idxs[nontarget_idxs] = nontarget_idxs[torch.argmax(dists[nontarget_idxs][:,nontarget_idxs], dim=-1)]
        neg_idxs[target_idxs] = nontarget_idxs[torch.argmin(dists[target_idxs][:,nontarget_idxs], dim=-1)]
        neg_idxs[nontarget_idxs] = target_idxs[torch.argmin(dists[nontarget_idxs][:,target_idxs], dim=-1)]
    return pos_idxs, neg_idxs


def reset_trainer(trainer):
    trainer.fit_loop.epoch_progress.reset()
    trainer.state.fn = TrainerFn.FITTING
    trainer.state.status = TrainerStatus.RUNNING
    trainer.training = True
    trainer.lightning_module.train()
    torch.set_grad_enabled(True)
    trainer.should_stop = False
    trainer.callbacks[0].wait_count = 0
    trainer.callbacks[0].stopped_epoch = 0
    trainer.callbacks[0].best_score = torch.tensor(torch.inf)
    trainer.callbacks[-1].best_model_score = None
    trainer.callbacks[-1].best_k_models = {}
    trainer.callbacks[-1].kth_best_model_path = ''
    trainer.callbacks[-1].kth_value = torch.tensor(torch.inf)
    trainer.callbacks[-1]._last_global_step_saved = 0
    trainer.callbacks[-1]._last_time_checked = None
    trainer.callbacks[-1].current_score = None
    trainer.callbacks[-1].best_model_path = ''

def update_xent(module, data_module, mult=1):
    p_target, p_nontarget = data_module.get_class_balance()
    n_target = int(p_target*len(data_module.data_train))
    n_nontarget = int(p_nontarget*len(data_module.data_train))
    n = len(data_module.data_train)
    target_weight = mult*n / (n_target*2) if n_target>0 else 1.0
    nontarget_weight = n / (n_nontarget*2) if n_nontarget>0 else 1.0
    module.criterion = torch.nn.NLLLoss(weight=torch.tensor([nontarget_weight, target_weight]))
    print(f'Auto target weight: {target_weight:.4f}')
    print(f'Auto nontarget weight: {nontarget_weight:.4f}')

def update_counts(module, data_module):
    n_target, n_nontarget = data_module.get_class_counts()
    module.n_target = n_target
    module.n_nontarget = n_nontarget
    print(n_target)
    print(n_nontarget)

def write_header(out_file, al_methods, ddm_exists=False):
    f = open(out_file, 'w')
    f.write('pass,time,train_time,train_epochs,inference_time,pre_dcf,pre_fnr,pre_fpr,dcf,fnr,fpr,pre_ns,pre_ps,ns,ps,pre_fns,pre_fps,fns,fps,diag_fns,diag_fps,p_target,p_nontarget,n_samples,cum_pre_dcf,cum_dcf,n_al,cf_tp,cf_fp,drift,model_zeros')
    if len(al_methods)==1 and al_methods[0]!='rand':
        f.write(',metric')
    f.write('\n')
    f.close()

def write_session(out_file, current_batch, test_results, error_counts, class_balance, n_samples, metric=None, 
                  drift_dist=None, n_al=0, cf_p=0, cf_n=0, has_drift=0, times=(0,0,0), epochs=0, model_zeros=0):
    elapsed_time, training_time, inference_time = times
    pre_fps, pre_fns, pre_ps, pre_ns, fps, fns, ps, ns, diag_fps, diag_fns, diag_ps, diag_ns = error_counts
    p_target, p_nontarget = class_balance
    f = open(out_file, 'a')
    f.write(f'{current_batch},{elapsed_time:.2f},{training_time:.2f},{epochs:d},{inference_time:.2f}')
    pre_fnr = pre_fns[-1]/pre_ps[-1] if pre_ps[-1]>0 else 0
    pre_fpr = pre_fps[-1]/pre_ns[-1] if pre_ns[-1]>0 else 0
    pre_dcf = 0.75*pre_fnr + 0.25*pre_fpr
    if test_results is not None:
        dcf = test_results[0]['test/dcf']
        fnr = test_results[0]['test/fnr']
        fpr = test_results[0]['test/fpr']
    else:
        dcf, fnr, fpr = 0, 0, 0
    f.write(f',{pre_dcf:.4f},{pre_fnr:.4f},{pre_fpr:.4f}')
    f.write(f',{dcf:.4f},{fnr:.4f},{fpr:.4f}')
    f.write(f',{pre_ns[-1]:d},{pre_ps[-1]:d}')
    f.write(f',{ns[-1]:d},{ps[-1]:d},{pre_fns[-1]:d},{pre_fps[-1]:d},{fns[-1]:d},{fps[-1]:d}')
    f.write(f',{diag_fns[-1]:d},{diag_fps[-1]:d}')
    f.write(f',{p_target:.4f},{p_nontarget:.4f},{n_samples:d}')
    pre_fnr = torch.sum(torch.LongTensor(pre_fns))/torch.sum(torch.LongTensor(ps)) if torch.sum(torch.LongTensor(ps))>0 else 0
    pre_fpr = torch.sum(torch.LongTensor(pre_fps))/torch.sum(torch.LongTensor(ns)) if torch.sum(torch.LongTensor(ns))>0 else 0
    cum_pre_dcf = 0.75*pre_fnr + 0.25*pre_fpr
    f.write(f',{cum_pre_dcf:.4f}')
    fnr = torch.sum(torch.LongTensor(fns))/torch.sum(torch.LongTensor(ps)) if torch.sum(torch.LongTensor(ps))>0 else 0
    fpr = torch.sum(torch.LongTensor(fps))/torch.sum(torch.LongTensor(ns)) if torch.sum(torch.LongTensor(ns))>0 else 0
    cum_dcf = 0.75*fnr + 0.25*fpr
    f.write(f',{cum_dcf:.4f},{n_al:d},{cf_p:d},{cf_n:d},{has_drift:d},{model_zeros:d}')
    if metric is not None:
        f.write(f',{metric:.4f}')
    f.write('\n')
    f.close()

def raw_score_gapfiller(data, threshold=0.0, kernsize=6):
    # :param scores: (time_steps,)
    # :param kernsize: int.
    #   default: int(gapcloser / shift) = int(33 / 5) = 6
    # :return data: (time_steps,) A vector of gapfilled raw scores.
    epsilon_to_push_boundary_points_above_threshold = 0.1

    ii = 0
    while ii < len(data):
        # Iterate until we find a segment that is not speech.
        if data[ii] >= threshold:
            ii += 1
            continue

        machine_segment_start = ii
        machine_segment_end = ii + 1

        # Iterate until we find a segment that is no longer machine.
        while machine_segment_end < len(data) and data[machine_segment_end] <= threshold:
            machine_segment_end += 1
            
        if machine_segment_end - machine_segment_start <= kernsize:
            # We want to pull up the intermediate non-speech values
            # to resemble something closer to a speech value.
            # We will shift the intermediate values up so that the minimum
            # is now at some epsilon. Then, we scale the values so that its
            # maximum equals whichever endpoint is largest. Finally, we set 
            # any value that ended up below that epsilon to be equal to that
            # epsilon.

            # N.B.: sanitized_machine_segment_end is only necessary here because we access
            # data[machine_segment_end] explicitly whereas we had been accessing it as a slice.
            # When slicing, the end index is taken to be end_index - 1 and we wouldn't
            # run into this problem.
            
            sanitized_machine_segment_end = machine_segment_end if machine_segment_end < len(data) else len(data) - 1
            new_intermediate_values = data[machine_segment_start:machine_segment_end] + torch.abs(torch.amin(data[machine_segment_start:machine_segment_end])) + epsilon_to_push_boundary_points_above_threshold
            new_intermediate_values /= torch.amax(new_intermediate_values)
            new_intermediate_values *= torch.amax([data[machine_segment_start], data[sanitized_machine_segment_end]])
            new_intermediate_values[new_intermediate_values < epsilon_to_push_boundary_points_above_threshold] = epsilon_to_push_boundary_points_above_threshold
            
            data[machine_segment_start:machine_segment_end] = new_intermediate_values

        ii = machine_segment_end
    
    return data