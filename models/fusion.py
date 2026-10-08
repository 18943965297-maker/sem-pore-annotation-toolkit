"""Gated fusion architecture and loss; expects externally supplied model logits."""

import torch

from torch import nn

import torch.nn.functional as F

def block(n, m):
    return nn.Sequential(nn.Conv2d(n,m,3,padding=1),nn.GroupNorm(4,m),nn.SiLU(),
                         nn.Conv2d(m,m,3,padding=1),nn.GroupNorm(4,m),nn.SiLU())

class EdgeSAMCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=block(7,16)
        self.enc2=block(16,32)
        self.mid=block(32,64)
        self.dec2=block(96,32)
        self.dec1=block(48,16)
        self.delta_head=nn.Conv2d(16,1,1)
        self.gate_head=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, image, ap_logits, ae_logits, edge_logits):
        ap=ap_logits.sigmoid(); ae=ae_logits.sigmoid()
        x=torch.cat((image,ap,ae,edge_logits.sigmoid(),(ap-ae).abs()),dim=1)
        f1=self.enc1(x); f2=self.enc2(F.avg_pool2d(f1,2))
        b=self.mid(F.avg_pool2d(f2,2))
        f=self.dec2(torch.cat((F.interpolate(b,size=f2.shape[-2:],mode='bilinear',align_corners=False),f2),1))
        f=self.dec1(torch.cat((F.interpolate(f,size=f1.shape[-2:],mode='bilinear',align_corners=False),f1),1))
        delta=self.delta_head(f); gate=self.gate_head(f).sigmoid()
        return {'logits':ap_logits+gate*delta,'gate':gate,'delta':delta}

def boundary(x):
    return F.max_pool2d(x,3,1,1)+F.max_pool2d(-x,3,1,1)

def loss_fn(logits,target,boundary_weight=.1):
    prob=logits.sigmoid();dims=(1,2,3)
    dice=1-((2*(prob*target).sum(dims)+1)/(prob.sum(dims)+target.sum(dims)+1)).mean()
    bce=F.binary_cross_entropy_with_logits(logits,target)
    bnd=F.l1_loss(boundary(prob),boundary(target))
    return bce+dice+boundary_weight*bnd
