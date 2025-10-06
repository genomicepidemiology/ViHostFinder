from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class HMNCF(nn.Module):

    def __init__(self, in_dims: int, hidsize_g: list, hidsize_l:list,
                 class_levels, names_levels, 
                  dropout, beta:float):

        super().__init__()
      
        self.class_metadata, self.total_pred = HMNCF.create_metadata(
                                                class_levels, names_levels,
                                                hidsize_g, hidsize_l)
        self.global_layers = nn.ModuleList()
        self.local_layers = nn.ModuleList()
        self.local_preds = nn.ModuleList()

        prev_hid = 0

        for idx, (k, val) in enumerate(self.class_metadata.items()):
            in_global = in_dims + prev_hid
            global_layer = nn.Sequential(
                    OrderedDict([
                        ("wg_{}".format(k), nn.Linear(in_global, val["hidsize_g"])),
                        ("ln_{}".format(k), nn.LayerNorm(val["hidsize_g"])),
                        ("act_{}".format(k), nn.ReLU()), 
                        ("dropout_{}".format(k), nn.Dropout(dropout))
                    ]))
            prev_hid = val["hidsize_g"]
            local_layer = nn.Sequential(
                    OrderedDict([
                        ("wl_{}".format(k), nn.Linear(val["hidsize_g"], val["hidsize_l"])),
                        ("ln_{}".format(k), nn.LayerNorm(val["hidsize_l"])),
                        ("act_{}".format(k), nn.ReLU()), 
                        ("dropout_{}".format(k), nn.Dropout(dropout))                        
                    ])
            )
            local_projection = nn.Sequential(
                    OrderedDict([
                        ("wpl_{}".format(k), nn.Linear(val["hidsize_l"], val["n_outputs"])),
                    ])
            )
            self.global_layers.append(global_layer)
            self.local_layers.append(local_layer)
            self.local_preds.append(local_projection)
        self.global_preds = nn.Sequential(
                    OrderedDict([
                        ("wpg_{}".format(k), nn.Linear(in_dims+prev_hid,
                                                       self.total_pred)),]))

        self.beta = beta
        print(self.global_layers.eval())

    @staticmethod
    def create_metadata(class_levels, names_levels, hidsize_g, hidsize_l):
        dict_data = OrderedDict()
        num_pred = 0
        for c in range(len(class_levels)):
            dict_data[names_levels[c]] = {"level":c,
                                          "n_outputs":class_levels[c],
                                          "hidsize_g":hidsize_g[c],
                                          "hidsize_l":hidsize_l[c]}
            num_pred += class_levels[c]
        return dict_data, num_pred


    def forward(self, x, device):
        out_global = torch.Tensor()
        out_global = out_global.to(device)
        local_predictions = []
        for global_layer, local_layer, local_pred in zip(self.global_layers, self.local_layers, self.local_preds):
            in_global = torch.cat([x,out_global], dim=1)
            out_global = global_layer(in_global)
            out_local = local_layer(out_global)
            pred_local = local_pred(out_local)

            local_predictions.append(pred_local)
        global_prediction = self.global_preds(torch.cat([x, out_global], dim=1))
        local_prediction = torch.cat(local_predictions, dim=1)

        final_prediction = self.beta*local_prediction + (1-self.beta)*global_prediction
        return final_prediction

    def loss_function(self, vanilla_loss, logits, labels, hier_restr=False,
                        sigma=0.5):
        loss_nn = vanilla_loss(logits, labels)
        loss_hier = 0
        if sigma != 0:
            for pair in hier_restr:
                loss_pair = logits[:,pair["Leave"]] - logits[:,pair["Parent"]]
                loss_pair = loss_pair[loss_pair>0]
                loss_pair = loss_pair**2
                loss_hier += sum(loss_pair)     # MEAN?
        return loss_nn + sigma*loss_hier

        
            
if __name__ == '__main__':
    num_batches = 4
    batch_size = 10
    hidden_state = 64
    fake_data = torch.randn((num_batches, batch_size, hidden_state))
    label_rows = {"Bacteria":0, "Plant":1, "Animalia":2, "Protist":3, "Cnidaria":4,
                  5: "Vertebrate", 6: "Invertebrate",
                  7: "Amphibia", 8: "Fish", 9: "Mammal", 10: "Bird", 11: "Reptile",
                  12: "Ecdysozoa", 13: "Spiralia", 14: "Human"}
    hier_cols = [{"Leave":"Human", "Parent":"Mammal"}, {"Leave":"Vertebrate", "Parent":"Mammal"},
                 {"Leave":"Vertebrate", "Parent":"Amphibia"}, {"Leave":"Vertebrate", "Parent":"Fish"},
                 {"Leave":"Vertebrate", "Parent":"Bird"}, {"Leave":"Vertebrate", "Parent":"Reptile"},
                 {"Leave":"Invertebrate", "Parent":"Ecdysozoa"}, {"Leave":"Invertebrate", "Parent":"Spiralia"}]
    parent_nums = []
    for col in hier_cols:
        parent_nums.append({"Leave": label_rows[col["Leave"]], 
                            "Parent": label_rows[col["Parent"]]})
    class_levels = [5, 2, 7, 1]


    hmncf = HMNCF(in_dims=hidden_state, hidsize_g=[128, 128, 256, 64],
                    hidsize_l=[64, 64, 128, 32], class_levels=[5, 2, 7, 1],
                    names_levels=["superkingdom", "order", "class", "specie"],
                    dropout=0.2,beta=0.5)
    vanilla_loss = nn.BCEWithLogitsLoss() 
    for b in range(num_batches):
        batch = fake_data[b, :, :]
        out = hmncf(batch)
        loss = hmncf.loss_function(vanilla_loss=vanilla_loss, logits=out, )