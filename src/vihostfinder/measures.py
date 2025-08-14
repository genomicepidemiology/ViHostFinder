from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, hamming_loss
from sklearn.metrics import log_loss, multilabel_confusion_matrix, precision_score, recall_score, roc_auc_score

class Metrics:

    @staticmethod
    def run_allmetrics(labels, preds, out):
        accuracy_score_v = accuracy_score(labels, preds)
        report = classification_report(labels,preds)
        hamming = hamming_loss(labels, preds)
        log_loss_v = log_loss(labels, preds)
        cm = multilabel_confusion_matrix(labels, preds)
        roc_auc_score_v = roc_auc_score(labels, out, average="samples")
        recall = recall_score(labels, preds, average="samples")
        precision= precision_score(labels,preds, average="samples")
        return (accuracy_score_v, report, hamming, log_loss_v, cm,
                    roc_auc_score_v, recall, precision)


