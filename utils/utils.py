import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus

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
    def __init__(self, fnr_weight=0.75, smax_weight=0):
        super(DcfLoss, self).__init__()
        self.alpha = fnr_weight
        self.beta = smax_weight
        self.smax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        '''
        x: Log of class posterior probabilities (on the interval [-inf,inf])
        y: Labels (0 or 1)
        '''
        x = torch.exp(x)
        if self.beta>0:
            x = self.smax(self.beta*x)
        expected_fnr = torch.dot(x[:,0],y*1.0)/torch.sum(y) if torch.sum(y)>0 else 0
        expected_fpr = torch.dot(x[:,1],(1.0-y))/torch.sum(1-y) if torch.sum(1-y)>0 else 0
        dcf = self.alpha*expected_fnr + (1-self.alpha)*expected_fpr
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

def update_xent(module, data_module):
    labels = torch.LongTensor(data_module.ds.labels)[torch.LongTensor(data_module.data_train.active_idxs)]
    n_target = torch.sum(labels)
    n_nontarget = len(labels) - n_target
    target_weight = len(labels) / (n_target*2)
    nontarget_weight = len(labels) / (n_nontarget*2)
    module.criterion = torch.nn.NLLLoss(weight=torch.tensor([nontarget_weight, target_weight]))