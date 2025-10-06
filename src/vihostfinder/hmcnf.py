from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class HMCN_old(nn.Module):
    def __init__(self,
                in_dims: int,
                hidden_size: int,
                classes, #: OrderedDict[str, Tuple[int, int]],  # ('class_name', (class_num, middle_size))
                act_fn: nn.Module = nn.ReLU,
                bias: bool = False,
                dropout: float = .5,
                beta: float = .5):
        super().__init__()

        self.classes_names = list(classes.keys())
        self.classes_nums = [i[0] for i in classes.values()]

        self.global_layers = nn.ModuleList()
        self.local_layers = nn.ModuleList()

        for i, (name, (cls_num, mid_size)) in enumerate(classes.items()):
            self.global_layers.append(
                nn.Sequential(OrderedDict([
                    (f'{name}_global_fc', nn.Linear(in_dims + (hidden_size if i else 0),
                                                    hidden_size, bias=bias)),
                    (f'{name}_global_act', act_fn()),
                    (f'{name}_global_ln', nn.LayerNorm(hidden_size)),
                    (f'{name}_global_dropout', nn.Dropout(dropout))
                    ]))
                )
            self.local_layers.append(
                nn.Sequential(OrderedDict([
                    (f'{name}_local_fc', nn.Linear(hidden_size, mid_size, bias=bias)),
                    (f'{name}_local_act', act_fn()),
                    (f'{name}_local_ln', nn.LayerNorm(mid_size)),
                    (f'{name}_local_fc2', nn.Linear(mid_size, cls_num, bias=bias))
                    ]))
                )

        self.proj_layer = nn.Linear(in_dims + hidden_size,
                                    sum([i[0] for i in classes.values()]),
                                    bias=bias)
        self.beta = beta

    def make_dict_logits(self, logits):
        return dict(zip(self.classes_names, torch.split(logits, self.classes_nums, dim=-1)))

    def forward(self,
                features: torch.Tensor,
                return_local_logits: bool = True,
                return_global_logits: bool = True,
                return_fused_logits: bool = True,
                return_logits_dict: bool = True):
        local_logits = []
        last_global = torch.Tensor().to(device=features.device)

        for global_layer, local_layer in zip(self.global_layers, self.local_layers):
            last_global = global_layer(torch.cat([features, last_global], dim=-1))
            local_logits.append(local_layer(last_global))

        local_logits = torch.cat(local_logits, dim=-1)
        global_logits = self.proj_layer(torch.cat([features, last_global], dim=-1))

        logits = {}

        if(return_local_logits):
            logits['local_logits'] = self.make_dict_logits(local_logits) if return_logits_dict else local_logits

        if(return_global_logits):
            logits['global_logits'] = self.make_dict_logits(global_logits) if return_logits_dict else global_logits

        if(return_fused_logits):
            fused_logits = local_logits * (1 - self.beta) + global_logits * self.beta
            logits['fused_logits'] = self.make_dict_logits(fused_logits) if return_logits_dict else fused_logits

        return logits

    def loss(self,
            inputs,
            targets,
            weights = None,
            reduction = 'mean',
            ignore_index: int = -100,
            hier_viol: float = .1):
        local_loss = {c: F.cross_entropy(inputs['local_logits'][c].transpose(-1, -2),
                                                targets[c],
                                                ignore_index=ignore_index,
                                                reduction=reduction) for c in self.classes_names}
        global_loss = {c: F.cross_entropy(inputs['global_logits'][c].transpose(-1, -2),
                                                targets[c],
                                                ignore_index=ignore_index,
                                                reduction=reduction) for c in self.classes_names}
        losses = {c + "_loss": (local_loss[c] + global_loss[c]) * \
                                    (weights[c] if weights is not None else 1) for c in self.classes_names}
        losses['total_loss'] = torch.stack(list(losses.values()))
        losses['total_loss'] = losses['total_loss'].mean() if reduction == 'mean' else losses['total_loss'].sum()

        if(hier_viol):
            preds = {c: F.softmax(inputs['fused_logits'][c], dim=-1) for c in self.classes_names}
            pred_masks = {c: targets[c] != ignore_index for c in self.classes_names}
            pred_scores = [(preds[c][torch.arange(preds[c].shape[0], device=preds[c].device)[:, None],
                                    torch.arange(preds[c].shape[1], device=preds[c].device)[None, :],
                                    targets[c]] * pred_masks[c]).mean() for c in self.classes_names]

            hier_viol_score = torch.stack([child - parent \
                                            for parent, child in zip(pred_scores[:-1], pred_scores[1:])])

            hier_viol_loss = F.relu(torch.max(hier_viol_score)) ** 2 * hier_viol
            losses['hierarchical_violation'] = hier_viol_loss
            losses['total_loss'] += hier_viol_loss

        return losses

    @staticmethod
    def run():
        CLASSES = OrderedDict([
                    ('class_a', (10, 16)),
                    ('class_b', (100, 192)),
                    ('class_c', (200, 512))])
        BATCH_SIZE = 2
        SAMPLE_NUM = 4

        classifier = HMCN_old(in_dims=512,
                        hidden_size=384,
                        classes=CLASSES)

        # (batch_size, sample_num, hidden_num)
        dummy_hiddens = torch.randn((BATCH_SIZE, SAMPLE_NUM, 512))

        logits = classifier(dummy_hiddens)

        print('fused_logits:')
        for k, v in logits['fused_logits'].items():
            print('    ', k, 'shape:', v.shape)

        dummy_targets = {name: torch.randint(0, cls_num, (BATCH_SIZE, SAMPLE_NUM)) \
                            for (name, (cls_num, mid_size)) in CLASSES.items()}

        losses = classifier.loss(logits, dummy_targets, ignore_index=0)

        print("losses:")
        for k, v in losses.items():
            print('   ', k+':', v.item())

        # You can just backward the total_loss!
        losses['total_loss'].backward()



# Example
if __name__ == '__main__':
    HMCN_old.run()