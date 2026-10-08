"""Trainable edgeSAM fusion; preserve AP and learn selection of Aedge additions."""
import torch
import torch.nn.functional as F
from .fusion import EdgeSAMCorrection, boundary


class EdgeSAMV2Supervised(EdgeSAMCorrection):
    def __init__(self, aedge_threshold=.3):
        super().__init__()
        self.aedge_threshold = aedge_threshold
        self.reset_addition_heads()

    def reset_addition_heads(self):
        torch.nn.init.zeros_(self.delta_head.weight)
        torch.nn.init.constant_(self.delta_head.bias, -8.)
        torch.nn.init.zeros_(self.gate_head.weight)
        torch.nn.init.zeros_(self.gate_head.bias)

    def forward(self, image, ap_logits, ae_logits, edge_logits, ap_mask=None, ae_mask=None):
        out = super().forward(image, ap_logits, ae_logits, edge_logits)
        ap = (ap_logits >= 0) if ap_mask is None else ap_mask.bool()
        ae = (ae_logits.sigmoid() >= self.aedge_threshold) if ae_mask is None else ae_mask.bool()
        candidate = ae & ~ap
        addition_logits = out['gate'] * out['delta']
        probability = ap.float() + candidate.float() * addition_logits.sigmoid()
        return dict(probability=probability, addition_logits=addition_logits,
                    candidate=candidate, ap_mask=ap, gate=out['gate'])


def supervised_loss(out, target):
    """Full v2 target is used; AP positives are already exactly retained by construction.

    Balance accepted/rejected Aedge candidates so sparse added pores are not ignored.
    No pseudo label is supplied to the inference network.
    """
    cand = out['candidate']; truth = target > .5
    pos = cand & truth; neg = cand & ~truth
    bce = F.binary_cross_entropy_with_logits(out['addition_logits'], target, reduction='none')
    terms = [bce[m].mean() for m in (pos, neg) if m.any()]
    balanced = torch.stack(terms).mean() if terms else bce.sum()*0
    q = out['addition_logits'].sigmoid() * cand.float()
    y = pos.float()
    add_dice = 1-(2*(q*y).sum()+1)/(q.sum()+y.sum()+1)
    p = out['probability']
    full_dice = 1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1)
    return balanced + add_dice + .25*full_dice + .1*F.l1_loss(boundary(p), boundary(target))
