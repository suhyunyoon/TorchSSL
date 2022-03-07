import torch
import torch.nn.functional as F
from train_utils import ce_loss, multilabel_sm_loss, multilabel_bce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value


def consistency_loss(logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2', 'multilabel_sm', 'multilabel_bce']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()
        # strong_prob, strong_idx = torch.max(torch.softmax(logits_s, dim=-1), dim=-1)
        # strong_select = strong_prob.ge(p_cutoff).long()
        # select = select * strong_select * (strong_idx == max_idx)
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()

    elif name == 'multilabel_bce':
        pseudo_label = torch.sigmoid(logits_w)

        # # Option 1 (same with single-label)
        # max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff).float().unsqueeze(dim=-1)
        # select = max_probs.ge(p_cutoff).long()
        
        # # Option 2 (thresholding minimum positive confidence)
        # pos_pl = pseudo_label.ge(0.5)
        # pos_pl = torch.where(pos_pl, pseudo_label, torch.ones(pseudo_label.size()).float().cuda(pseudo_label.device))
        # min_pos_probs, max_idx = torch.min(pos_pl, dim=-1)
        # mask = min_pos_probs.ge(p_cutoff).float().unsqueeze(dim=-1)
        # select = min_pos_probs.ge(p_cutoff).long()

        # # Option 3 (masking ambigous each classes(0.5<=logit<p_cutoff))
        # max_probs, max_idx = torch.max(pseudo_label, dim=-1) # useless
        # mask_bool = torch.logical_or(pseudo_label.lt(0.5), pseudo_label.ge(p_cutoff))
        # mask, select = mask_bool.float(), mask_bool.long()

        # Option 3 (mask each class(logit>=p_cutoff))
        max_probs, max_idx = torch.max(pseudo_label, dim=-1) # useless
        mask_bool = pseudo_label.ge(p_cutoff)
        mask, select = mask_bool.float(), mask_bool.long()

        if use_hard_labels:
            pseudo_hard_label = pseudo_label.ge(0.5).float()
            masked_loss = multilabel_bce_loss(logits_s, pseudo_hard_label, use_hard_labels, reduction='none') * mask
        else:
            # useless
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = multilabel_bce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        
        return masked_loss.mean(), mask.mean(), select, max_idx.long()

    else:
        assert Exception('Not Implemented consistency_loss')
            