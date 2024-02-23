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
        self.dist_func = euclidean_distance
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
    trainer.callbacks[3].best_model_score = None
    trainer.callbacks[3].best_k_models = {}
    trainer.callbacks[3].kth_best_model_path = ''
    trainer.callbacks[3].kth_value = torch.tensor(torch.inf)
    trainer.callbacks[3]._last_global_step_saved = 29
    trainer.callbacks[3]._last_time_checked = None
    trainer.callbacks[3].current_score = None
    trainer.callbacks[3].best_model_path = ''

def update_xent(module, data_module):
    p_target, p_nontarget = data_module.get_class_balance()
    target_weight = 1 / (p_target*2)
    nontarget_weight = 1 / (p_nontarget*2)
    module.criterion = torch.nn.NLLLoss(weight=torch.tensor([nontarget_weight, target_weight]))

def write_header(out_file, al_methods, ddm_exists):
    f = open(out_file, 'w')
    f.write('pass,pre_dcf,pre_fnr,pre_fpr,dcf,fnr,fpr,ns,ps,pre_fns,pre_fps,fns,fps,p_target,p_nontarget,n_samples,cum_pre_dcf,cum_dcf')
    if len(al_methods)==1 and al_methods[0]!='rand':
        f.write(',metric')
    if ddm_exists:
        f.write(',drift')
    f.write('\n')
    f.close()

def write_session(out_file, current_batch, test_results, error_counts, class_balance, n_samples, metric, drift_dist):
    pre_fps, pre_fns, fps, fns, ps, ns = error_counts
    p_target, p_nontarget = class_balance
    f = open(out_file, 'a')
    f.write(f'{current_batch}')
    pre_fnr = pre_fns[-1]/ps[-1] if ps[-1]>0 else 0
    pre_fpr = pre_fps[-1]/ns[-1] if ns[-1]>0 else 0
    pre_dcf = 0.75*pre_fnr + 0.25*pre_fpr
    dcf = test_results[0]['test/dcf']
    fnr = test_results[0]['test/fnr']
    fpr = test_results[0]['test/fpr']
    f.write(f',{pre_dcf:.4f},{pre_fnr:.4f},{pre_fpr:.4f}')
    f.write(f',{dcf:.4f},{fnr:.4f},{fpr:.4f}')
    f.write(f',{ns[-1]:d},{ps[-1]:d},{pre_fns[-1]:d},{pre_fps[-1]:d},{fns[-1]:d},{fps[-1]:d}')
    f.write(f',{p_target:.4f},{p_nontarget:.4f},{n_samples:d}')
    pre_fnr = torch.sum(torch.LongTensor(pre_fns))/torch.sum(torch.LongTensor(ps)) if torch.sum(torch.LongTensor(ps))>0 else 0
    pre_fpr = torch.sum(torch.LongTensor(pre_fps))/torch.sum(torch.LongTensor(ns)) if torch.sum(torch.LongTensor(ns))>0 else 0
    cum_pre_dcf = 0.75*pre_fnr + 0.25*pre_fpr
    f.write(f',{cum_pre_dcf:.4f}')
    fnr = torch.sum(torch.LongTensor(fns))/torch.sum(torch.LongTensor(ps)) if torch.sum(torch.LongTensor(ps))>0 else 0
    fpr = torch.sum(torch.LongTensor(fps))/torch.sum(torch.LongTensor(ns)) if torch.sum(torch.LongTensor(ns))>0 else 0
    cum_dcf = 0.75*fnr + 0.25*fpr
    f.write(f',{cum_dcf:.4f}')
    if metric is not None:
        f.write(f',{metric:.4f}')
    if drift_dist is not None:
        f.write(f',{drift_dist:.4f}')
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