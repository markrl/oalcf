import torch
import torch.nn.functional as F
import numpy as np

from pdb import set_trace

class FeedbackSimulator:
    def __init__(self, params):
        self.sim_type = params.sim_type
        self.max_fb_samples = params.max_fb_samples

    def simulate(self, data_module, module):
        if self.sim_type is None:
            return []
        idxs = []
        if self.sim_type=='all':
            idxs.append(torch.arange(data_module.unlabeled_len()))
        else:
            preds, scores = self.extract_preds(data_module, module)
            labels = data_module.get_test_labels()
            if 'fps' in self.sim_type:
                idx = np.where(np.logical_and(labels==0, preds==1))[0]
                if len(idx)==0:
                    idxs.append([])
                elif len(idx)==1:
                    idxs.append([idx[0]])
                else:
                    score = scores[idx]
                    idxs.append(idx[np.argsort(1-score)])
            if 'tps' in self.sim_type:
                idx = np.where(np.logical_and(labels==1, preds==1))[0]
                if len(idx)==0:
                    idxs.append([])
                elif len(idx)==1:
                    idxs.append([idx[0]])
                else:
                    score = scores[idx]
                    idxs.append(idx[np.argsort(score-0.5)])
            if 'fns' in self.sim_type:
                idx = np.where(np.logical_and(labels==1, preds==0))[0]
                if len(idx)==0:
                    idxs.append([])
                elif len(idx)==1:
                    idxs.append([idx[0]])
                else:
                    score = scores[idx]
                    idxs.append(idx[np.argsort(1-score)])
            if 'tns' in self.sim_type:
                idx = np.where(np.logical_and(labels==0, preds==0))[0]
                if len(idx)==0:
                    idxs.append([])
                elif len(idx)==1:
                    idxs.append([idx[0]])
                else:
                    score = scores[idx]
                    idxs.append(idx[np.argsort(score-0.5)])
        if len(idxs) == 0:
            return []
        idxs = self.interleave(idxs)

        if self.max_fb_samples is None:
            return idxs
        else:
            return idxs[:self.max_fb_samples]

    def extract_preds(self, data_module, module):
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
        scores, preds = torch.max(F.softmax(logits, dim=-1), dim=-1)
        return preds, scores

    def interleave(self, list_list):
        out = []
        list_list = [list(ll)[::-1] for ll in list_list if len(ll)>0]
        while len(list_list) > 0:
            for ll in list_list:
                out.append(ll.pop())
            list_list = [ll for ll in list_list if len(ll)>0]
        return out