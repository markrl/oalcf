import torch
import torch.nn as nn
import numpy as np
from skactiveml.classifier import ParzenWindowClassifier, SklearnClassifier
from river.compat import River2SKLClassifier
from river.forest import ARFClassifier, AMFClassifier

from pdb import set_trace

class ArfModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = ARFClassifier()

    def forward(self, x):
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

