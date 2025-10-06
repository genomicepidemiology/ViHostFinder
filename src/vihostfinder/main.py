import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import argparse
import os
from datetime import datetime

from vihostfinder.arguments import VArguments
from vihostfinder.data_utils import HostDataset, ClusterSampler
from vihostfinder.model import MultiLabelFFNN
from vihostfinder.hmnc_f import HMNCF
from vihostfinder.measures import Metrics, HierMetrics
from vihostfinder.output import OutputModule, WANDBModule

class ViralInf:

    BATCH_SIZE = 64
    BOOTSTRAP_SIZE = 10

    def __init__(self, outfolder, lm="hyenadna", prediction_type="orig_hier1"):
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if lm == "hyenadna":
            self.input_dim = 256
        else:
            self.input_dim = 256
        self._metadata = {}
        self._metadata["lm"] = lm
        self._metadata["input_dim"] = self.input_dim
        self.prediction_dict = self.set_labels(prediction=prediction_type)

        self.output_module = OutputModule(outfolder=outfolder)

    def set_labels(self, prediction):
        prediction_dict = {}

        if prediction == "orig_flat":
        
            prediction_dict["class_num"] = 13
            prediction_dict["class_levels"] = None
            prediction_dict["class_names"] = None
            prediction_dict["labels"] = list(HostDataset.orig_all_hosts.keys())
            prediction_dict["hierarchical_order"] = None
        
        elif prediction == "orig_hier1":
        
            prediction_dict["labels"] = list(HostDataset.orig_hier1_hosts.keys())
            prediction_dict["class_num"] = len(prediction_dict["labels"])
            prediction_dict["class_levels"] = [6, 2, 7, 1]
            prediction_dict["class_names"] = ["superkingdom", "order", "class", "specie"]
            prediction_dict["hierarchical_order"] = HostDataset.create_tree(labels=prediction)

            assert sum(prediction_dict["class_levels"]) == prediction_dict["class_num"]
            assert len(prediction_dict["class_levels"]) == len(prediction_dict["class_names"])
        
        elif prediciton == "orig_hier2":
            classes = 20
        else:
            raise ValueError("The prediction mode {} does not exists".format(prediciton))
        self._metadata["label_dict"] = prediction_dict
        self._metadata["prediction_type"] = prediction
        return prediction_dict

    def set_model(self, model_type="flatNN", hidsize_global=[128, 128, 256, 64],
                    hidsize_local=[64, 64, 128, 32], dropout=0.2, beta=0.5):
        self._metadata["model_name"] = model_type
        self._metadata["dropout"] = dropout
        if model_type == "flatNN":
            self.model = MultiLabelFFNN(input_dim=self.input_dim, output_dim=self.prediction_dict["class_num"],
                                        dropout_rate=dropout)
            self.model_name = model_type
        elif model_type == "HMCNF":
            self.model = HMNCF(in_dims=self.input_dim, hidsize_g=hidsize_global,
                    hidsize_l=hidsize_local, class_levels=self.prediction_dict["class_levels"],
                    names_levels=self.prediction_dict["class_names"],
                    dropout=dropout, beta=beta)
            self.model_name = model_type
            self._metadata["hidsize_global"] = hidsize_global
            self._metadata["hidsize_local"] = hidsize_local
            self._metadata["beta"] = beta

        else:
            ValueError("The model {} is not set".format(model_type))


    def set_train_optimizers(self, lr=0.001, sigma=0.2):

        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigma = sigma
        self._metadata["lr"] = lr
        self._metadata["sigma"] = sigma


    def train_epoch(self, train_loader):
        self.model.train()
        len_dataloader = len(train_dataloader)
        train_loss = 0.0
        for idx, (inputs, labels) in tqdm(enumerate(train_loader)):

            inputs, labels = inputs.to(self.device), labels.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs, device=self.device)
            if self.model_name == "flatNN":
                loss = self.criterion(outputs, labels)
            elif self.model_name == "HMCNF":
                loss = self.model.loss_function(vanilla_loss=self.criterion, logits=outputs,
                                                labels=labels, hier_restr=self.prediction_dict["hierarchical_order"],
                                                sigma=self.sigma)
            loss.backward()
            self.optimizer.step()
            if self.wandb_report:
                self.wandb_report.add_step_info(loss.item(), lr=self._metadata["lr"], batch_n=idx,
                                                    len_dataloader=len_dataloader)
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
                outputs = self.model(inputs, device=self.device)
                if self.model_name == "simpleNN":
                    loss = self.criterion(outputs, labels)
                elif self.model_name == "HMCNF":
                    loss = self.model.loss_function(vanilla_loss=self.criterion, logits=outputs,
                                                    labels=labels, hier_restr=self.prediction_dict["hierarchical_order"],
                                                    sigma=self.sigma)

                val_loss += loss.item()
                out_sig = torch.sigmoid(outputs)
                preds = out_sig > 0.5
                
                out_lst.append(out_sig.detach().cpu().numpy())
                pred_lst.append(preds.detach().cpu().numpy())
                labels_lst.append(labels.detach().cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        preds_t = np.concatenate(pred_lst)
        labels_t = np.concatenate(labels_lst)
        out_t = np.concatenate(out_lst)
        return avg_val_loss, preds_t, out_t, labels_t

    def _set_metrics(self, labels, outs):
        metrics = Metrics(labels=labels, out_cont=outs)          
        report, cm, df_results = metrics.run_allmetrics()
        hier_metrics = HierMetrics(labels=labels, out_cont=outs,
                                    labels_names=self.prediction_dict["labels"])
        acc_perlabel = hier_metrics.acc_per_label()
        acc_perlevel = hier_metrics.acc_per_level()
        #hier_precision, hier_recall, hier_f1 = hier_metrics.hierarchical_prec_recall_f1(
         #                                           y_true=labels, y_pred=outs, 
          #                                          hierarchy=self.prediction_dict[], num_labels)
        return report, cm, df_results, acc_perlabel, acc_perlevel

    def loadevaluate_loaders(self, annotations_file, file_dir, partition, cluster_freq=None,
                                sampler_cluster=False):
        eval_dataloader = ViralInf.create_loader(annotations_file=annotations_file,
                                        file_dir=file_dir, partition=partition, labels=self.prediction_dict["labels"])
        eval_sample1_dataloader = ViralInf.create_loader(annotations_file=annotations_file,
                                        file_dir=file_dir, partition=val_part, labels=self.prediction_dict["labels"],
                                        sampler=True, cluster_freq=0.)
        
        if sample_cluster:
            eval_sample2_dataloader = ViralInf.create_loader(annotations_file=annotations_file,
                                file_dir=file_dir, partition=val_part, labels=self.prediction_dict["labels"],
                                sampler=sampler_cluster, cluster_freq=cluster_frequency)
        else:
            eval_sample2_dataloader = None

        return eval_dataloader, eval_sample1_dataloader, eval_sample2_dataloader

    @staticmethod
    def bootstrap_df(df_lst):
        stacked_results = np.stack([df.values for df in df_lst])
        mean_df = pd.DataFrame(np.mean(stacked_results, axis=0), columns=df_lst[0].columns)
        std_df = pd.DataFrame(np.std(stacked_results, axis=0), columns=df_lst[0].columns)
        return mean_df, std_df

    def evaluate(self, eval_data, bootstrap_mode=False, fraq_clust=0., cluster_column="",
                    name_partition="", wandb_report=False):
        dataloader = DataLoader(eval_data, batch_size=ViralInf.BATCH_SIZE)
        avg_val_loss, preds_t, out_t, labels_t = self.val_epoch(dataloader)
        preds_df = pd.DataFrame(preds_t, columns=self.prediction_dict["labels"])
        outs_df = pd.DataFrame(out_t, columns=self.prediction_dict["labels"])
        labels_df = pd.DataFrame(labels_t, columns=self.prediction_dict["labels"])
        results_lst = []
        perlabel_lst = []
        perlevel_lst = []
        if not bootstrap_mode:
            report, cm, df_results, acc_perlabel, acc_perlevel = self._set_metrics(labels_df, outs_df)
            results_lst.append(df_results)
            perlabel_lst.append(acc_perlabel)
            perlevel_lst.append(acc_perlevel) 
            bootstrap_mode = "NoBootstrap"
           
        elif bootstrap_mode == "random":
            for i in range(ViralInf.BOOTSTRAP_SIZE):
                n_samples = len(labels_df)
                bootstrap_indices = np.random.choice(labels_df.index, size=n_samples, replace=True)
                report, cm, df_results, acc_perlabel, acc_perlevel = self._set_metrics(labels_df.iloc[bootstrap_indices], outs_df.iloc[bootstrap_indices])
                results_lst.append(df_results)
                perlabel_lst.append(acc_perlabel)
                perlevel_lst.append(acc_perlevel)

        elif bootstrap_mode == "clusters":
            for i in range(ViralInf.BOOTSTRAP_SIZE):
                sampler = ClusterSampler(dataset=eval_data, cluster_column=cluster_column, f=fraq_clust)
                report, cm, df_results, acc_perlabel, acc_perlevel = self._set_metrics(labels_df.iloc[sampler], outs_df.iloc[sampler])
                results_lst.append(df_results)
                perlabel_lst.append(acc_perlabel)
                perlevel_lst.append(acc_perlevel)

        results_mean, results_std = ViralInf.bootstrap_df(results_lst)
        perlabel_mean, perlabel_std = ViralInf.bootstrap_df(perlabel_lst)
        perlevel_mean, perlevel_std = ViralInf.bootstrap_df(perlevel_lst)
        if self.wandb_report:
            overall_dict = WANDBModule.format_metrics(dict_metrics=df_results.iloc[0].to_dict(),
                                                        prefix="Evaluate/{}/{}/Overall Metrics".format(bootstrap_mode, name_partition))
            label_dict = WANDBModule.format_metrics(dict_metrics=acc_perlabel,
                                                        prefix="Evaluate/{}/{}/Label Metrics".format(bootstrap_mode, name_partition))
            level_dict = WANDBModule.format_metrics(dict_metrics=acc_perlevel,
                                                        prefix="Evaluate/{}/{}/Level Metrics".format(bootstrap_mode, name_partition))
            combined_metrics= {**overall_dict, **label_dict, **level_dict}
            self.wandb_report.add_info(combined_metrics)            
        return results_mean, results_std, perlabel_mean, perlabel_std, perlevel_mean, perlevel_std

    def train(self, train_loader, val_loader, epochs=10, wandb_report=False):
        self._metadata["epochs"]  = epochs
        if wandb_report:
            self.wandb_report = WANDBModule(name=os.path.basename(self.output_module.folder_path),
                                    configuration=self._metadata,
                                    wandb_dir=self.output_module.folder_path)
            self.wandb_report.start_train_report(model=self.model, criterion=self.criterion)
        else:
            self.wandb_report = False

        self.model.to(self.device)
        preds_lst = []
        outs_lst = []
        labels_lst = []
        measures = []
        for epoch in range(epochs):

            avg_train_loss = self.train_epoch(train_loader=train_loader)
            avg_val_loss, preds, outs, labels = self.val_epoch(val_loader=val_loader)

            report, cm, df_results, acc_perlabel, acc_perlevel = self._set_metrics(labels, outs)
            print("Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}".format(
                   epoch=epoch+1, epochs=epochs, avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss))
            print("Accuracy:{acc} | Hamming Loss:{hamm} | Log Loss:{logloss} | ROCAUC:{rocauc} | Recall:{recall} | Precision:{prec}".format(
                   acc=df_results["SubsetAccuracy"], hamm=df_results["HammingLoss"], logloss=df_results["LogLoss"], rocauc=df_results["ROCAUC (Average)"],
                   recall=df_results["Recall (Average)"], prec=df_results["Precision (Average)"]))
            df_results["Epoch"] = epoch
            df_results["TrainLoss"] = avg_train_loss
            df_results["ValLoss"] = avg_val_loss
            if self.wandb_report:
                overall_dict = WANDBModule.format_metrics(dict_metrics=df_results.iloc[0].to_dict(),
                                                            prefix="Overall Metrics")
                label_dict = WANDBModule.format_metrics(dict_metrics=acc_perlabel,
                                                            prefix="Label Metrics")
                level_dict = WANDBModule.format_metrics(dict_metrics=acc_perlevel,
                                                            prefix="Level Metrics")
                combined_metrics= {**overall_dict, **label_dict, **level_dict, **{"Epoch":epoch}}
                self.wandb_report.add_epoch_info(combined_metrics)
            measures.append(df_results)
            preds_lst.append(np.expand_dims(preds, axis=0))
            outs_lst.append(np.expand_dims(outs, axis=0))
            labels_lst.append(np.expand_dims(labels, axis=0))
        preds_np = np.concatenate(preds_lst)
        outs_np = np.concatenate(outs_lst)
        labels_np = np.concatenate(labels_lst)
        measures_df = pd.concat(measures)

        return measures_df, preds_np, outs_np, labels_np

    def loadtraining_loaders(self, annotations_file, file_dir, sampler_cluster,
                                cluster_frequency, train_part, val_part):
        train_dataloader = ViralInf.create_loader(annotations_file=annotations_file,
                                file_dir=file_dir, partition=train_part, labels=self.prediction_dict["labels"],
                                sampler=sampler_cluster, cluster_freq=cluster_frequency)

        val_dataloader = ViralInf.create_loader(annotations_file=annotations_file,
                                file_dir=file_dir, partition=val_part, labels=self.prediction_dict["labels"],
                                sampler=sampler_cluster, cluster_freq=cluster_frequency)
        
        self._metadata["val_partition"] = val_part
        self._metadata["train_partition"] = train_part
        self._metadata["sampler_cluster"] = sampler_cluster
        self._metadata["cluster_frequency"] = cluster_frequency
        self._metadata["file_annotation"] = annotations_file

        return train_dataloader, val_dataloader

    @staticmethod
    def create_loader(annotations_file, file_dir, partition, labels, sampler, cluster_freq):
        data = HostDataset(annotations_file=annotations_file,
                                file_dir=file_dir, partition=partition, labels=labels)
        if sampler:
            sampler = ClusterSampler(data, cluster_column=sampler, f=cluster_freq)
            dataloader = DataLoader(data, batch_size=ViralInf.BATCH_SIZE,  sampler=sampler)
        else:
            dataloader = DataLoader(data, batch_size=ViralInf.BATCH_SIZE, shuffle=True)
        return dataloader

    def make_results(self, outfolder, measures, preds, outs, labels):
        self.output_module.save_training_results(measures=measures, preds=preds, outs=outs, labels=labels)
        self.output_module.save_nn(model=self.model)


if __name__ == "__main__":

    vargs = VArguments.create_arguments()

    vargs.global_layers = VArguments.fix_layers(vargs.global_layers)
    vargs.local_layers = VArguments.fix_layers(vargs.local_layers)
    train_part, val_part, test_part = VArguments.select_partitions(partition=vargs.partition)
    file_dir = VArguments.select_filedir(lm=vargs.lm, sampler=vargs.sampler_cluster)
    if vargs.annotation_file:
        annotations_file = vargs.annotation_file
    else:
        annotations_file = VArguments.get_annotationfile(sampler=vargs.sampler_cluster)

    viralmodel = ViralInf(outfolder=vargs.out, lm=vargs.lm, prediction_type=vargs.labels)

    viralmodel.set_model(model_type=vargs.model, hidsize_global=vargs.global_layers,
                        hidsize_local=vargs.local_layers, dropout=vargs.dropout, beta=vargs.beta)

    if vargs.command == "train":

        train_dataloader, val_dataloader = viralmodel.loadtraining_loaders(annotations_file=annotations_file, file_dir=file_dir,
                                    sampler_cluster=vargs.sampler_cluster,
                                    cluster_frequency=vargs.cluster_frequency, train_part=train_part, val_part=val_part)
        viralmodel.set_train_optimizers(lr=vargs.lr, sigma=vargs.sigma)
        measures, preds, outs, labels = viralmodel.train(train_dataloader, val_dataloader, epochs=int(vargs.epochs),
                                                            wandb_report=vargs.wandb)
        val_data = HostDataset(annotations_file=annotations_file,
                                file_dir=file_dir, partition=val_part, labels=viralmodel.prediction_dict["labels"])
        viralmodel.evaluate(eval_data=val_data, name_partition="Validation", wandb_report=vargs.wandb)
        viralmodel.evaluate(eval_data=val_data, bootstrap_mode="random", name_partition="Validation", wandb_report=vargs.wandb)
        if vargs.sampler_cluster:
            viralmodel.evaluate(eval_data=val_data, bootstrap_mode="clusters",name_partition="Validation", fraq_clust=0., cluster_column=vargs.sampler_cluster,wandb_report=vargs.wandb)
        test_data = HostDataset(annotations_file=annotations_file,
                                file_dir=file_dir, partition=test_part, labels=viralmodel.prediction_dict["labels"])
        viralmodel.evaluate(eval_data=test_data, name_partition="Test", wandb_report=vargs.wandb)
        viralmodel.evaluate(eval_data=test_data, bootstrap_mode="random", name_partition="Test", wandb_report=vargs.wandb)
        if vargs.sampler_cluster:
            viralmodel.evaluate(eval_data=val_data, bootstrap_mode="clusters",fraq_clust=0., name_partition="Test", cluster_column=vargs.sampler_cluster, wandb_report=vargs.wandb)

        viralmodel.make_results(outfolder=vargs.out,preds=preds, outs=outs, labels=labels, measures=measures)

        viralmodel.wandb_report.finish_report()


