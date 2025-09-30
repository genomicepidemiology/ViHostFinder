from sklearn.metrics import accuracy_score, classification_report, hamming_loss, f1_score
from sklearn.metrics import log_loss, multilabel_confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.metrics import coverage_error, average_precision_score, label_ranking_loss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tree_utils import TreeOutput

class Metrics:

    def __init__(self, labels, out_cont, threshold=0.5):

        self.labels = labels
        self.outCont = out_cont
        self.preds = (self.outCont >= threshold).astype(int)

    @staticmethod
    def calc_subsetaccuracy(labels, preds):
        accuracy_score_v = accuracy_score(labels, preds)
        return accuracy_score_v
    
    @staticmethod
    def calc_hammingloss(labels, preds):
        hamming = hamming_loss(labels, preds)
        return hamming
    
    @staticmethod
    def calc_logloss(labels, outs):
        logloss = log_loss(labels, outs)
        return logloss
    
    @staticmethod
    def calc_coverage(labels, outs):
        cov = coverage_error(labels, outs)
        return cov
    
    @staticmethod
    def calc_rankingloss(labels, outs):
        rankloss = label_ranking_loss(labels, outs)
        return rankloss
    
    @staticmethod
    def calc_recall(labels, preds, mode):
        recall = recall_score(labels, preds, average=mode)
        return recall
    
    @staticmethod
    def calc_precision(labels, preds, mode):
        precision= precision_score(labels,preds, average=mode)
        return precision
    
    @staticmethod
    def calc_f1(labels, preds, mode):
        f1 = f1_score(y_true=labels, y_pred=preds, average=mode)
        return f1

    @staticmethod
    def calc_averageprecision(labels, outs, mode):
        avg_prec = average_precision_score(labels, outs, average=mode)
        return avg_prec
    
    @staticmethod
    def roc_auc_score(labels, outs, mode):
        rcauc = roc_auc_score(labels, outs, average=mode)
        return rcauc

   
    def run_allmetrics(self):

        subacc = Metrics.calc_subsetaccuracy(labels=self.labels, preds=self.preds)
        hamm = Metrics.calc_hammingloss(labels=self.labels, preds=self.preds)
        logloss = Metrics.calc_logloss(labels=self.labels, outs=self.outCont)
        rankingloss = Metrics.calc_rankingloss(labels=self.labels, outs=self.outCont)
        coverage = Metrics.calc_coverage(labels=self.labels, outs=self.outCont)
        recallavg = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="samples")
        recallweig = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="weighted")
        recallmicro = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="micro")
        recallmacro = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="macro")
        precavg = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="samples")
        precweig = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="weighted")
        precmicro = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="micro")
        precmacro = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="macro")
        f1avg = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="samples")
        f1weig = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="weighted")
        f1micro = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="micro")
        f1macro = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="macro")
        avgprecavg = Metrics.calc_averageprecision(labels=self.labels, outs=self.outCont, mode="samples")
        avgprecweig = Metrics.calc_averageprecision(labels=self.labels, outs=self.outCont, mode="weighted")
        avgprecmicro = Metrics.calc_averageprecision(labels=self.labels, outs=self.outCont, mode="micro")
        avgprecmacro = Metrics.calc_averageprecision(labels=self.labels, outs=self.outCont, mode="macro")
        rocaucavg = Metrics.roc_auc_score(labels=self.labels, outs=self.outCont, mode="samples")
        rocaucweig = Metrics.roc_auc_score(labels=self.labels, outs=self.outCont, mode="weighted")
        rocaucmicro = Metrics.roc_auc_score(labels=self.labels, outs=self.outCont, mode="micro")
        rocaucmacro = Metrics.roc_auc_score(labels=self.labels, outs=self.outCont, mode="macro")
        df_results = pd.DataFrame([[subacc, hamm, logloss, rankingloss, coverage, 
                                    recallavg, recallmicro, recallmacro, recallweig,
                                    precavg, precmicro, precmacro, precweig,
                                    f1avg, f1micro, f1macro, f1weig,
                                    avgprecavg, avgprecmicro, avgprecmacro, avgprecweig,
                                    rocaucavg, rocaucmicro, rocaucmacro, rocaucweig]],
                                   columns=["SubsetAccuracy", "HammingLoss", "LogLoss", "RankingLoss","Coverage",
                                            "Recall (Average)", "Recall (Micro)","Recall (Macro)", "Recall (Weighted)",
                                            "Precision (Average)","Precision (Micro)", "Precision (Macro)", "Precision (Weighted)",
                                            "F1 (Average)","F1 (Micro)","F1 (Macro)", "F1 (Weighted)",
                                            "AveragePrecision (Average)","AveragePrecision (Micro)","AveragePrecision (Macro)", "AveragePrecision (Weighted)",
                                            "ROCAUC (Average)", "ROCAUC (Micro)", "ROCAUC (Macro)", "ROCAUC (Weighted)"])

        report = classification_report(self.labels,self.preds)
        cm = multilabel_confusion_matrix(self.labels, self.preds)
        return  report, cm, df_results

    @staticmethod
    def lineplot(dict_lines, outname, ylabel="Loss", epochs=None):
        df = pd.DataFrame(dict_lines)        
        df_long = df.melt(var_name="Category", value_name=ylabel)
        df_long["Epochs"] = df_long.groupby("Category").cumcount()
        sns.lineplot(data=df_long, x="Epochs", y=ylabel, hue="Category")
        plt.savefig(outname)
        plt.close()

    @staticmethod
    def barplot(dict_measures, measure_name, outname):
        pass

    @staticmethod
    def tree_plot(npz_file):
        print(npz_file["labels"].shape)
        print(npz_file["preds"].shape)
        print(npz_file["outs"].shape)
        last_out = npz_file["outs"][-1,:,:]
        last_labels = npz_file["labels"][-1,:,:]
        last_preds = npz_file["preds"][-1,:,:]

        labels = ["Bacteria","Plant", "Protist", "Cnidaria", "Fungi",
                "Amphibia", "Fish", "Mammal", "Bird", "Reptile", "Ecdysozoa", "Spiralia",
                "Human"]
        label_rows = {"Bacteria":0, "Plant":1, "Animalia":2, "Protist":3, "Cnidaria":4, "Fungi":5,
                 "Vertebrate":6, "Invertebrate":7,
                 "Amphibia": 8, "Fish":9, "Mammal":10, "Bird":11, "Reptile":12,
                 "Ecdysozoa": 13, "Spiralia": 14, "Human":15}
        label_map = ["Bacteria","Plant", "Animalia", "Protist", "Cnidaria", "Fungi",
                        "Vertebrate","Invertebrate",
                        "Amphibia", "Fish", "Mammal", "Bird", "Reptile", "Ecdysozoa", "Spiralia",
                        "Human"]
        tree_baseline = TreeOutput()
        tree_baseline.save_tree("test/graphs/blank.png")            
        for l in labels:

            lab_subl = last_labels[last_labels[:,label_rows[l]]==1,:]
            out_subl = last_out[last_labels[:,label_rows[l]]==1,:]
            lab_mean = np.mean(lab_subl, axis=0)
            out_mean = np.mean(out_subl, axis=0)
            lab_std = np.std(lab_subl, axis=0)
            out_std = np.std(out_subl, axis=0)
            output_vector = []
            for mean, std in zip(lab_mean, lab_std):
                output_vector.append((mean,std))
            tree_baseline = TreeOutput()
            tree_baseline.fillpred_tree(output_vector, label_map)
            tree_baseline.save_tree("test/graphs/label{}.png".format(l))            

            output_vector = []
            for mean, std in zip(out_mean, out_std):
                output_vector.append((mean,std))
            tree_baseline = TreeOutput()
            tree_baseline.fillpred_tree(output_vector, label_map)
            tree_baseline.save_tree("test/graphs/out{}.png".format(l)) 


        


if __name__ == "__main__":
    example_csv = pd.read_csv("../../ViralInf/results_trainembed2/partition1hmcnfb0s0.csv",
                                sep="\t")
    Metrics.lineplot({"TrainLoss":example_csv["TrainLoss"], "ValLoss":example_csv["ValLoss"]},
                    outname="test/graphs/loss_lineplot_hmncfb0s0.png")
    Metrics.lineplot({"SubsetAccuracy":example_csv["SubsetAccuracy"]},
                    outname="test/graphs/acc_lineplot_hmncfb0s0.png", ylabel="Accuracy")
    example_csv = pd.read_csv("../../ViralInf/results_trainembed2/partition1hmcnfb05s0.csv",
                                sep="\t")
    Metrics.lineplot({"TrainLoss":example_csv["TrainLoss"], "ValLoss":example_csv["ValLoss"]},
                    outname="test/graphs/loss_lineplot_hmncfb05s0.png")
    Metrics.lineplot({"SubsetAccuracy":example_csv["SubsetAccuracy"]},
                    outname="test/graphs/acc_lineplot_hmncfb05s0.png", ylabel="Accuracy")
    example_csv = pd.read_csv("../../ViralInf/results_trainembed2/partition1hmcnfb0s05.csv",
                                sep="\t")
    Metrics.lineplot({"TrainLoss":example_csv["TrainLoss"], "ValLoss":example_csv["ValLoss"]},
                    outname="test/graphs/loss_lineplot_hmncfb0s05.png")
    Metrics.lineplot({"SubsetAccuracy":example_csv["SubsetAccuracy"]},
                    outname="test/graphs/acc_lineplot_hmncfb0s05.png", ylabel="Accuracy")
    example_csv = pd.read_csv("../../ViralInf/results_trainembed2/partition1hmcnfb10s1.csv",
                                sep="\t")
    Metrics.lineplot({"TrainLoss":example_csv["TrainLoss"], "ValLoss":example_csv["ValLoss"]},
                    outname="test/graphs/loss_lineplot_hmncfb10s1.png")
    Metrics.lineplot({"SubsetAccuracy":example_csv["SubsetAccuracy"]},
                    outname="test/graphs/acc_lineplot_hmncfb10s1.png", ylabel="Accuracy")
    example_csv = pd.read_csv("../../ViralInf/results_trainembed2/partition1hmcnfb05s1.csv",
                                sep="\t")
    Metrics.lineplot({"TrainLoss":example_csv["TrainLoss"], "ValLoss":example_csv["ValLoss"]},
                    outname="test/graphs/loss_lineplot_hmncfb05s1.png")
    Metrics.lineplot({"SubsetAccuracy":example_csv["SubsetAccuracy"]},
                    outname="test/graphs/acc_lineplot_hmncfb05s1.png", ylabel="Accuracy")
    example_csv = pd.read_csv("../../ViralInf/results_trainembed2/partition1hmcnfb10s0.csv",
                                sep="\t")
    Metrics.lineplot({"TrainLoss":example_csv["TrainLoss"], "ValLoss":example_csv["ValLoss"]},
                    outname="test/graphs/loss_lineplot_hmncfb10s0.png")
    Metrics.lineplot({"SubsetAccuracy":example_csv["SubsetAccuracy"]},
                    outname="test/graphs/acc_lineplot_hmncfb10s0.png", ylabel="Accuracy")

    example_npz= np.load("../../ViralInf/results_trainembed2/partition1hmcnfb05s0.npz")
    Metrics.tree_plot(example_npz)

