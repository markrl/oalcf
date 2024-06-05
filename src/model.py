import torch
import torch.nn as nn
from audlib.nn.nn import MLP

class ClassModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        dim = 0
        for rr in params.feat_root.split(','):
            if 'ecapa' in rr:
                dim += 256
            elif 'wavlm' in rr:
                dim += 1024
            else:
                dim += 512
        dim *= params.context*2+1

        if 'seq' in params.feat_root:
            self.rnn = nn.GRU(dim, int(dim/2), params.rnn_layers, batch_first=True, dropout=params.drop, bidirectional=True)
        else:
            self.rnn = None

        self.bn_layer = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(params.drop)
        self.mlp = MLP(dim, 2, hiddims=[params.class_hidden]*(params.class_layers+1),
                    activate_hid=nn.LeakyReLU(),
                    activate_out=None,
                    batchnorm=[True]*(params.class_layers+1))
        self.activate_out = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        x = self.drop(x)
        if not self.params.no_initial_bn:
            x = self.bn_layer(x)
        if self.rnn is not None:
            x = self.rnn(x)[0]
            x = torch.mean(x, dim=1)
        logits = self.mlp(x)
        return self.activate_out(x), logits


class CompModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        dim = 0
        for rr in params.feat_root.split(','):
            if 'ecapa' in rr:
                dim += 256
            elif 'wavlm' in rr:
                dim += 1024
            else:
                dim += 512
        dim *= params.context*2+1

        self.bn_layer = nn.BatchNorm1d(dim)
        # self.bn_layer = nn.InstanceNorm1d(dim)
        self.mlp = MLP(dim, params.comp_hidden, hiddims=[params.comp_hidden]*params.comp_layers,
                    activate_hid=nn.LeakyReLU(),
                    activate_out=nn.Linear(params.comp_hidden, params.embed_dim),
                    batchnorm=[True]*params.comp_layers)

    def forward(self, x):
        if not self.params.no_initial_bn:
            x = self.bn_layer(x)
        return self.mlp(x)


class CompClassModel(CompModel):
    def __init__(self, params):
        super().__init__(params)
        dim = 0
        for rr in params.feat_root.split(','):
            if 'ecapa' in rr:
                dim += 256
            elif 'wavlm' in rr:
                dim += 1024
            else:
                dim += 512
        dim *= params.context*2+1
            
        if 'seq' in params.feat_root:
            self.rnn = nn.GRU(dim, int(dim/2), params.rnn_layers, batch_first=True, dropout=params.drop, bidirectional=True)
        else:
            self.rnn = None
        self.classifier = MLP(params.embed_dim, 2, hiddims=[params.class_hidden]*params.class_layers,
                    activate_hid=nn.LeakyReLU(),
                    activate_out=None,
                    batchnorm=[True]*params.class_layers)

        self.activate_out = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        if not self.params.no_initial_bn:
            x = self.bn_layer(x)
        if self.rnn is not None:
            x = self.rnn(x)[0]
            x = torch.mean(x, dim=1)
        embeds = self.mlp(x)
        logits = self.classifier(embeds)
        return embeds, self.activate_out(logits), logits