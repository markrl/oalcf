import torch
import torch.nn as nn
from audlib.nn.nn import MLP

class Autoencoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        dim = 0
        for rr in params.feat_root.split(','):
            dim += 512 if 'wavlm' not in rr else 1024
        dim *= params.context*2+1

        self.bn_layer = nn.BatchNorm1d(dim)
        self.mlp = MLP(dim, params.comp_hidden, hiddims=[params.comp_hidden]*params.comp_layers,
                    activate_hid=nn.LeakyReLU(),
                    activate_out=nn.Linear(params.comp_hidden, params.embed_dim),
                    batchnorm=[True]*params.comp_layers)
        # Currently, the final two layers of the normal classifier (batchnorm and linear) are cut off
        self.classifier = MLP(params.embed_dim, params.class_hidden, hiddims=[params.class_hidden]*(params.class_layers-1),
                    activate_hid=nn.LeakyReLU(),
                    activate_out=None,
                    batchnorm=[True]*params.class_layers)

        self.dec = MLP(params.class_hidden, dim, hiddims=[params.class_hidden, params.class_hidden*2, params.class_hidden*4],
                    activate_hid=nn.LeakyReLU(),
                    activate_out=None,
                    batchnorm=[True]*params.class_layers)

    def forward(self, x):
        if not self.params.no_initial_bn:
            x = self.bn_layer(x)
        comp_embeds = self.mlp(x)
        class_embeds = self.classifier(comp_embeds)
        decode_out = self.dec(class_embeds)
        return decode_out


        