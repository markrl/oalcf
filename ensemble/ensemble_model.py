import torch
import torch.nn as nn
import numpy as np
from river.forest import ARFClassifier, AMFClassifier
from river.compat import River2SKLClassifier

from pdb import set_trace

class ArfModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        n_estimators = 10
        self.model = ARFClassifier(n_estimators)
        if 'corpus' in params.__dict__:
            self.model = River2SKLClassifier(self.model)
            self.stream = False
        else:
            self.stream = True

    def forward(self, x):
        if not self.stream:
            if self.training:
                x,y = x
                dev = x.device
                x = x.cpu()
                y = y.cpu()
                self.model.partial_fit(x, y, [0,1])
                set_trace()
            else:
                dev = x.device
                x = x.cpu()
            out = self.model.predict_proba(x)
            out = torch.from_numpy(out)
        else:
            if self.training:
                x,y = x
                dev = x.device
                x = x.cpu()
                y = y.cpu()
                for sample,label in zip(x,y):
                    sample_dict = {ii:float(xx) for ii,xx in enumerate(sample)}
                    self.model.learn_one(sample_dict, int(label))
            else:
                dev = x.device
                x = x.cpu()
            out = torch.empty(len(x),2)
            for ii,sample in enumerate(x):
                sample_dict = {ii:xx for ii,xx in enumerate(sample)}
                prob = self.model.predict_proba_one(sample_dict)
                for kk in prob:
                    out[ii,kk] = prob[kk]
        
        return out.to(dev)