import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


    # def plot_confusion_matrix(self, class_names=None):
    #     """Plots the confusion matrix using Matplotlib"""
    #     if class_names is None:
    #         class_names = [f'Class {i}' for i in range(self.n_classes)]
        
    #     cm = self.confusion_matrix
    #     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    #     plt.figure(figsize=(10, 8))
    #     disp = ConfusionMatrixDisplay(cm_normalized, display_labels=class_names)
    #     disp.plot(include_values=True, xticks_rotation='vertical', values_format='.2f')
    #     plt.title("Normalized Confusion Matrix")
    #     plt.xlabel("Predicted Class")
    #     plt.ylabel("True Class")
    #     plt.show()



    def plot_confusion_matrix(self, class_names = None):
        
        fig, ax = plt.subplots(figsize=(8, 6.7), dpi=300)
        
        if class_names is None:
            class_names = [f'{i}' for i in range(self.n_classes)]
        
        cm = self.confusion_matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        
        sns.heatmap(cm_normalized, fmt = '.2f', annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 8})

        # ax.set_title(f"Normalized Confusion Matrix", fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        return fig




    def get_precision_recall_f1(self):
        tp = self.confusion_matrix.diagonal()
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'Precision': precision.mean(),
            'Recall': recall.mean(),
            'F1-Score': f1.mean(),
            'Confusion Matrix': self.confusion_matrix
            }

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


# import numpy as np
# from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
# import matplotlib.pyplot as plt
# import seaborn as sns

# class _StreamMetrics(object):
#     def __init__(self):
#         """ Overridden by subclasses """
#         raise NotImplementedError()

#     def update(self, gt, pred):
#         """ Overridden by subclasses """
#         raise NotImplementedError()

#     def get_results(self):
#         """ Overridden by subclasses """
#         raise NotImplementedError()

#     def to_str(self, metrics):
#         """ Overridden by subclasses """
#         raise NotImplementedError()

#     def reset(self):
#         """ Overridden by subclasses """
#         raise NotImplementedError()      

# class StreamSegMetrics(_StreamMetrics):
#     """
#     Stream Metrics for Semantic Segmentation Task
#     """
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.confusion_matrix = np.zeros((n_classes, n_classes))
#         self.all_targets = []
#         self.all_preds = []

#     def update(self, label_trues, label_preds):
#         for lt, lp in zip(label_trues, label_preds):
#             self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
#             self.all_targets.extend(lt.flatten())
#             self.all_preds.extend(lp.flatten())

#     @staticmethod
#     def to_str(results):
#         string = "\n"
#         for k, v in results.items():
#             if k != "Class IoU":
#                 string += "%s: %f\n" % (k, v)
#         return string

#     def _fast_hist(self, label_true, label_pred):
#         mask = (label_true >= 0) & (label_true < self.n_classes)
#         hist = np.bincount(
#             self.n_classes * label_true[mask].astype(int) + label_pred[mask],
#             minlength=self.n_classes ** 2,
#         ).reshape(self.n_classes, self.n_classes)
#         return hist

#     def get_results(self):
#         """Returns accuracy score evaluation result.
#             - overall accuracy
#             - mean accuracy
#             - mean IU
#             - fwavacc
#         """
#         hist = self.confusion_matrix
#         acc = np.diag(hist).sum() / hist.sum()
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#         acc_cls = np.nanmean(acc_cls)
#         iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#         mean_iu = np.nanmean(iu)
#         freq = hist.sum(axis=1) / hist.sum()
#         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#         cls_iu = dict(zip(range(self.n_classes), iu))

#         return {
#             "Overall Acc": acc,
#             "Mean Acc": acc_cls,
#             "FreqW Acc": fwavacc,
#             "Mean IoU": mean_iu,
#             "Class IoU": cls_iu,
#         }

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
#         self.all_targets = []
#         self.all_preds = []

#     def plot_confusion_matrix(self, class_names=None):
#         fig, ax = plt.subplots(figsize=(8, 6.7), dpi=300)

#         if class_names is None:
#             class_names = [f'{i}' for i in range(self.n_classes)]

#         cm = self.confusion_matrix
#         cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#         sns.heatmap(cm_normalized, fmt='.2f', annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 8})

#         ax.set_xlabel('Predicted Label', fontsize=12)
#         ax.set_ylabel('True Label', fontsize=12)

#         plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
#         plt.setp(ax.get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")
#         plt.tight_layout()
#         return fig

#     def get_precision_recall_f1(self):
#         tp = self.confusion_matrix.diagonal()
#         fp = self.confusion_matrix.sum(axis=0) - tp
#         fn = self.confusion_matrix.sum(axis=1) - tp

#         precision = tp / (tp + fp + 1e-10)
#         recall = tp / (tp + fn + 1e-10)
#         f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

#         return {
#             'Precision': precision.mean(),
#             'Recall': recall.mean(),
#             'F1-Score': f1.mean(),
#             'Confusion Matrix': self.confusion_matrix
#         }

# class AverageMeter(object):
#     """Computes average values"""
#     def __init__(self):
#         self.book = dict()

#     def reset_all(self):
#         self.book.clear()
    
#     def reset(self, id):
#         item = self.book.get(id, None)
#         if item is not None:
#             item[0] = 0
#             item[1] = 0

#     def update(self, id, val):
#         record = self.book.get(id, None)
#         if record is None:
#             self.book[id] = [val, 1]
#         else:
#             record[0] += val
#             record[1] += 1

#     def get_results(self, id):
#         record = self.book.get(id, None)
#         assert record is not None
#         return record[0] / record[1]
