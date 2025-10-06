import os
import torch
from datetime import datetime
import wandb
import numpy as np


class OutputModule:

    def __init__(self, outfolder):
        now = datetime.now()
        suffix = now.strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.abspath("{}__{}".format(outfolder, suffix))
        os.mkdir(folder_path)
        self.folder_path = folder_path

    def save_training_results(self, measures, preds, outs, labels):
        np.savez("{}/trainingresults.npz".format(self.folder_path), preds=preds, outs=outs, labels=labels)
        measures.to_csv("{}/measures.csv".format(self.folder_path), sep="\t", index=False)
    
    def save_nn(self, model):
        torch.save(model.state_dict(), "{}/model_weights.pth".format(self.folder_path))
    


class WANDBModule:
    
    batch_checkpoint = 15

    def __init__(self, name, configuration={}, wandb_dir=None):

        # Start a new wandb run to track this script.
        self.wandb_run = wandb.init(entity="epidemiology_dl", project="VirInFinder",
                                config=configuration, dir=wandb_dir,
                                name=name)


    def start_train_report(self, model, criterion, log="all"):
        self.wandb_run.watch(model, criterion, log="all", log_freq=1)
        model_params = self.count_params(model)
        self.wandb_run.summary["Trainable parameters"] = model_params
        self.step_wandb = 0
        self.epoch = 0

    def count_params(self, model):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params

    def add_epoch_info(self, log_results):
        self.epoch = log_results["Epoch"]
        wandb.log(log_results, step=self.step_wandb)

    def add_info(self, log_results):
        wandb.log(log_results)

    def add_step_info(self, loss_train, lr, batch_n, len_dataloader):
        if batch_n % WANDBModule.batch_checkpoint == 1 and self.wandb_run:
            wandb.log({"Training Loss/Step": loss_train, "Learning Rate": lr, "Epoch": self.epoch + ((batch_n+1)/len_dataloader)}, step=self.step_wandb)
            self.step_wandb += 1

    def finish_report(self):
        self.wandb_run.finish()

    def log_plot(self, fig, name):
        self.wandb_run.log({name: fig})

    def bar_plot(self, title, table):
        barplot = wandb.plot.bar(table, "Label", "Quantity", title=title)
        wandb.log({"Fixed Barplot": barplot})


    @staticmethod
    def format_metrics(dict_metrics, prefix):
        metrics = {}

        if isinstance(dict_metrics, dict):
            for k, v in dict_metrics.items():
                metrics[f"{prefix}/{k}"] = v
        else:
            for idx in dict_metrics.index:
                for col in dict_metrics.columns:
                    metrics[f"{prefix}/{idx}/{col}"] = dict_metrics.at[idx, col]

        return metrics



