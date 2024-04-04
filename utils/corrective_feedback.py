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
        elif self.sim_type=='all':
            idxs = torch.arange(data_module.unlabeled_len())
        elif self.sim_type=='fps':
            preds = self.extract_preds(data_module, module)
            labels = data_module.get_test_labels()
            idxs = np.where(np.logical_and(labels==0, preds==1))[0]
        elif self.sim_type=='fpstps':
            preds = self.extract_preds(data_module, module)
            labels = data_module.get_test_labels()
            idxs = np.where(preds==1)[0]

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
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        return preds