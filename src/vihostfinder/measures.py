from sklearn.metrics import accuracy_score, classification_report, hamming_loss, f1_score
from sklearn.metrics import log_loss, multilabel_confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.metrics import coverage_error, average_precision_score, label_ranking_loss, matthews_corrcoef
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
        if ylabel == "Loss":
            plt.ylim(0, 0.4)
        else:
            plt.ylim(0,1)
        plt.savefig(outname)
        plt.close()

    @staticmethod
    def barplot(dict_measures, measure_name, outname):
        pass

    @staticmethod
    def tree_plot(npz_file):
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

class HierMetrics:

    OrigLabels = {"FirstLevel": ["Bacteria", "Plant", "Animalia", "Protist", "Cnidaria", "Fungi"],
                  "SecondLevel": ["Vertebrate", "Invertebrate"],
                  "ThirdLevel": ["Amphibia", "Fish", "Mammal", "Bird", "Reptile", "Spiralia", "Ecdysozoa"],
                  "FourthLevel": ["Human"]}
    
    def __init__(self, labels, out_cont, labels_names, threshold=0.5):

        self.labels_df = pd.DataFrame(labels, columns=labels_names)
        self.outCont_df = pd.DataFrame(out_cont, columns=labels_names)
        self.predictions_df = (self.outCont_df >= threshold).astype(int)
        self.labels_names = labels_names

    def acc_per_label(self):
        prec = []
        recall = []
        mcc = []
        f1 = []
        for l in self.labels_names:
            mcc.append(matthews_corrcoef(self.labels_df[l], self.predictions_df[l]))
            recall.append(recall_score(self.labels_df[l], self.predictions_df[l]))
            prec.append(precision_score(self.labels_df[l], self.predictions_df[l]))
            f1.append(f1_score(self.labels_df[l], self.predictions_df[l]))
        df_results = pd.DataFrame(list(zip(prec, recall, mcc, f1)), columns=["Precision", "Recall", "MCC", "F1"], index=self.labels_names)
        return df_results
    
    def acc_per_level(self):
        results = {}
        for k in HierMetrics.OrigLabels.keys():
            lvl_labels = self.labels_df[HierMetrics.OrigLabels[k]]
            lvl_preds = self.predictions_df[HierMetrics.OrigLabels[k]]
            results[k] = {}
            prec_macro = Metrics.calc_precision(lvl_labels, lvl_preds, mode="macro")
            prec_weighted = Metrics.calc_precision(lvl_labels, lvl_preds, mode="weighted")
            recall_macro = Metrics.calc_recall(lvl_labels, lvl_preds, mode="macro")
            recall_weighted = Metrics.calc_recall(lvl_labels, lvl_preds, mode="weighted")
            f1_macro = Metrics.calc_f1(lvl_labels, lvl_preds, mode="macro")
            f1_weighted = Metrics.calc_f1(lvl_labels, lvl_preds, mode="weighted")
            hamming = Metrics.calc_hammingloss(lvl_labels, lvl_preds)
            subsetacc = Metrics.calc_subsetaccuracy(lvl_labels, lvl_preds)
            results[k]["Precision (macro)"] = prec_macro
            results[k]["Precision (weighted)"] = prec_weighted
            results[k]["Recall (macro)"] = recall_macro
            results[k]["Recall (weighted)"] = recall_weighted
            results[k]["F1 (macro)"] = f1_macro
            results[k]["F1 (weighted)"] = f1_weighted
            results[k]["Hamming"] = hamming
            results[k]["Subset Accuracy"] = subsetacc
        df_results = pd.DataFrame.from_dict(results, orient='index')
        return df_results

    @staticmethod
    def build_ancestor_map(hierarchy, num_labels):
        ancestor_map = {i: set() for i in range(num_labels)}
        parent_map = {}

        for relation in hierarchy:
            leave = relation["Leave"]
            parent = relation["Parent"]
            parent_map[leave] = parent

        for i in range(num_labels):
            current = i
            while current in parent_map:
                parent = parent_map[current]
                ancestor_map[i].add(parent)
                current = parent

        return ancestor_map


    def hierarchical_prec_recall_f1(self, y_true, y_pred, hierarchy, num_labels):
        ancestor_map = HierMetrics.build_ancestor_map(hierarchy=hierarchy, num_labels=num_labels)
        def expand_with_ancestors(indices):
            expanded = set(indices)
            for idx in indices:
                expanded.update(ancestor_map.get(idx, set()))
            return expanded

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        n_samples = len(y_true)

        for true_indices, pred_indices in zip(y_true, y_pred):
            true_expanded = expand_with_ancestors(true_indices)
            pred_expanded = expand_with_ancestors(pred_indices)

            intersection = true_expanded & pred_expanded
            precision = len(intersection) / len(pred_expanded) if pred_expanded else 0.0
            recall = len(intersection) / len(true_expanded) if true_expanded else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        avg_precision = total_precision / n_samples
        avg_recall = total_recall / n_samples
        avg_f1 = total_f1 / n_samples

        return avg_precision, avg_recall, avg_f1



        


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

