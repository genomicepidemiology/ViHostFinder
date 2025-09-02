import torch
from tqdm import tqdm
from data_utils import HostDataset
from model import MultiLabelFFNN
from hmnc_f import HMNCF
from measures import Metrics
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import json
import numpy as np
import argparse

class ViralInf:

    def __init__(self, lr=0.001, lm="hyenadna", dropout=0.2, prediction="main",
                        device="cuda", weights=None, model="simpleNN"):
        self.device = device
        if prediction == "main":
            classes = 7
            class_levels = [7]
        else:
            classes = 15
            class_levels = [6, 2, 7, 1]
            hidsize_g = [128, 128, 256, 64]
            hidsize_l = [64, 64, 128, 32]
            names_levels = ["superkingdom", "order", "class", "specie"]
        if lm == "hyenadna":
            input_dim = 256
        else:
            input_dim = 256
        if model == "simpleNN":
            self.model = MultiLabelFFNN(input_dim=input_dim, output_dim=classes,
                                        dropout_rate=dropout)
            self.model_name = model
        elif model == "HMCNF":
            self.model = HMNCF(in_dims=input_dim, hidsize_g=hidsize_g,
                    hidsize_l=hidsize_l, class_levels=class_levels,
                    names_levels=names_levels,
                    dropout=0.2,beta=0.5)
            self.model_name = model
            self.sigma = 0.3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()


    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader):

            inputs, labels = inputs.to(self.device), labels.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        return avg_train_loss
    
    def val_epoch(self, val_loader):
        # Validation
        self.model.eval()
        val_loss = 0.0
        pred_lst = []
        labels_lst = []
        out_lst = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                outputs = self.model(inputs)
                if self.model_name == "simpleNN":
                    loss = self.criterion(outputs, labels)
                elif self.model_name == "HMCNF":
                    loss = self.model.loss_function(vanilla_loss=self.criterion, logits=outputs,
                                                    labels=labels, hier_restr=False,
                                                    sigma=self.sigma)

                val_loss += loss.item()
                out_sig = torch.sigmoid(outputs)
                preds = out_sig > 0.5
                
                out_lst.append(out_sig.detach().cpu())
                pred_lst.append(preds.detach().cpu())
                labels_lst.append(labels.detach().cpu())
        avg_val_loss = val_loss / len(val_loader)
        preds_t = torch.concat(pred_lst)
        labels_t = torch.concat(labels_lst)
        out_t = torch.concat(out_lst)
        return avg_val_loss, preds_t, out_t, labels_t


    def train(self, train_loader, val_loader, epochs=10):
        self.model.to(self.device)
        measures = {"trainloss":[], "valloss":[], "accuracy":[], "hamming":[],"logloss":[],
                "recall":[], "precision":[], "rocauc":[], "preds":[], "outs":[], "labels":[]}
        preds_lst = []
        outs_lst = []
        labels_lst = []

        for epoch in range(epochs):

            avg_train_loss = self.train_epoch(train_loader=train_loader)
            avg_val_loss, preds, outs, labels = self.val_epoch(val_loader=val_loader)
            (accuracy_score, report, hamming, log_loss, cm,
                        roc_auc_score, recall, precision) = Metrics.run_allmetrics(labels, preds, outs)
            print("Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}".format(
                                epoch=epoch+1, epochs=epochs, avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss))
            print("Accuracy:{acc} | Hamming Loss:{hamm} | Log Loss:{logloss} | ROCAUC:{rocauc} | Recall:{recall} | Precision:{prec}".format(
                                acc=accuracy_score, hamm=hamming, logloss=log_loss, rocauc=roc_auc_score, recall=recall, prec=precision))
            measures["trainloss"].append(avg_train_loss)
            measures["valloss"].append(avg_val_loss)
            measures["accuracy"].append(accuracy_score)
            measures["rocauc"].append(roc_auc_score)
            measures["recall"].append(recall)
            measures["precision"].append(precision)
            measures["logloss"].append(log_loss)
            measures["hamming"].append(hamming)
            preds_lst.append(np.expand_dims(preds, axis=0))
            outs_lst.append(np.expand_dims(outs, axis=0))
            labels_lst.append(np.expand_dims(labels, axis=0))
        preds_np = np.concatenate(preds_lst)
        outs_np = np.concatenate(outs_lst)
        labels_np = np.concatenate(labels_lst)
        return measures, preds_np, outs_np, labels_np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--lm', help='lm', choices=["hyenadna", "caduceus"])
    parser.add_argument('--labels', help='labels', choices=["main","all"])
    parser.add_argument('--partition', help='labels', choices=["1","2","3","4","5"])
    parser.add_argument("--model", help="model", choices=["simpleNN", "HMCNF"])
    parser.add_argument("--out")
    parser.add_argument("--epochs")
    args = parser.parse_args()

    if args.lm == "hyenadna":
        file_dir = "/work3/alff/ViralInf/RNA_db/data/embeddingvector/embeddingvector_hyenadna/"
    else:
        file_dir = "/work3/alff/ViralInf/RNA_db/data/embeddingvector/embeddingvector_caduceus/"

    if args.partition == 1:
        train_part = [1,2,3]
        val_part = [4]
    elif args.partition == 2:
        train_part = [2,3,4]
        val_part = [5]
    elif args.partition == 3:
        train_part = [3,4,5]
        val_part = [1]
    elif args.partition == 4:
        train_part = [4,5,1]
        val_part = [2]
    else:
        train_part = [5,1,2]
        val_part = [3]

    training_data = HostDataset(annotations_file="/work3/alff/ViralInf/RNA_db/data/metadata/repr_dbredux_part.tsv",
                                file_dir=file_dir, partition=train_part, labels=args.labels)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    val_data = HostDataset(annotations_file="/work3/alff/ViralInf/RNA_db/data/metadata/repr_dbredux_part.tsv",
                                file_dir=file_dir, partition=val_part, labels=args.labels)

    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

    viralmodel = ViralInf(prediction=args.labels, lm=args.lm)
    measures, preds, outs, labels = viralmodel.train(train_dataloader, val_dataloader, epochs=int(args.epochs))
    outfile = args.out
    with open("{}.json".format(outfile), 'w') as f:
        json.dump(measures, f)
    np.savez("{}.npz".format(outfile), preds=preds, outs=outs, labels=labels)


