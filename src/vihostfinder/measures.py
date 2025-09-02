from sklearn.metrics import accuracy_score, classification_report, hamming_loss, f1_score
from sklearn.metrics import log_loss, multilabel_confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.metrics import coverage_error, average_precision_score, label_ranking_loss
import pandas as pd

class Metrics:

    def __init__(self, labels, out_cont, threshold=0.5):

        self.labels = labels
        self.outCont = out_cont
        self.predictions = (self.outCont >= threshold).astype(int)

    @staticmethod
    def calc_subsetaccuracy(labels, preds):
        accuracy_score_v = accuracy_score(labels, preds)
        return accuracy_score_v
    
    @staticmethod
    def calc_hammingloss(labels, preds):
        hamming = hamming_loss(labels, preds)
        return hamming
    
    @staticmethod
    def calc_logloss(labels, preds):
        logloss = log_loss(labels, preds)
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
        logloss = Metrics.calc_logloss(labels=self.labels, preds=self.outCont)
        rankingloss = Metrics.calc_rankingloss(labels=self.labels, preds=self.outCont)
        recallavg = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="average")
        recallmicro = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="micro")
        recallmacro = Metrics.calc_recall(labels=self.labels, preds=self.preds, mode="macro")
        precavg = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="average")
        precmicro = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="micro")
        precmacro = Metrics.calc_precision(labels=self.labels, preds=self.preds, mode="macro")
        f1avg = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="average")
        f1micro = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="micro")
        f1macro = Metrics.calc_f1(labels=self.labels, preds=self.preds, mode="macro")
        avgprecavg = Metrics.calc_averageprecision(labels=self.labels, preds=self.preds, mode="average")
        avgprecmicro = Metrics.calc_averageprecision(labels=self.labels, preds=self.preds, mode="micro")
        avgprecmacro = Metrics.calc_averageprecision(labels=self.labels, preds=self.preds, mode="macro")
        rocaucavg = Metrics.roc_auc_score(labels=self.labels, preds=self.preds, mode="average")
        rocaucmicro = Metrics.roc_auc_score(labels=self.labels, preds=self.preds, mode="micro")
        rocaucmacro = Metrics.roc_auc_score(labels=self.labels, preds=self.preds, mode="macro")
        df_results = pd.DataFrame([subacc, hamm, logloss, rankingloss, recallavg, recallmicro, recallmacro,
                                   precavg, precmicro, precmacro, f1avg, f1micro, f1macro, avgprecavg,
                                   avgprecmicro, avgprecmacro, rocaucavg, rocaucmicro, rocaucmacro],
                                   columns=["SubsetAccuracy", "HammingLoss", "LogLoss", "RankingLoss",
                                            "Recall (Average)", "Recall (Micro)","Recall (Maccro)",
                                            "Precision (Average)","Precision (Micro)", "Precision (Maccro)",
                                            "F1 (Average)","F1 (Micro)","F1 (Maccro)",
                                            "AveragePrecision (Average)","AveragePrecision (Micro)","AveragePrecision (Maccro)",
                                            "ROCAUC (Average)", "ROCAUC (Micro)", "ROCAUC (Macro)"])

        report = classification_report(self.labels,self.preds)
        cm = multilabel_confusion_matrix(self.labels, self.preds)
        return  report, cm, df_results


