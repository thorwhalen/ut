__author__ = 'thor'

"""
Utilities to measure binary classification performance based on the confusion matrix.
Definitions taken from http://en.wikipedia.org/wiki/Confusion_matrix.

Author: Thor Whalen
"""

from numpy import *
import sklearn as sk
import matplotlib.pyplot as plt


# metric_mat: dict of matrices which produce specific count metrics when "scalar multiplied" with a confusion matrix
metric_mat = {
    'tp': array([[0, 0], [0, 1]]),
    'tn': array([[1, 0], [0, 0]]),
    'fp': array([[0, 1], [0, 0]]),
    'fn': array([[0, 0], [1, 0]])
}
metric_mat.update({
    'p': metric_mat['tp'] + metric_mat['fn'],
    'n': metric_mat['tn'] + metric_mat['fp'],
})
metric_mat.update({
    'total': metric_mat['p'] + metric_mat['n']
})

# rate_metric_mats: dict of pairs of matrices to produce specific "rate metrics"
# Both elements of the pair should be "scalar multiplied" (i.e. sum(A * B) in numpy) by the confusion matrix,
# and then the first result divided by the second to get the rate.
rate_metric_mats = {
    'recall': (metric_mat['tp'], metric_mat['p']),
    'specificity': (metric_mat['tn'], metric_mat['n']),
    'precision': (metric_mat['tp'], metric_mat['tp'] + metric_mat['fp']),
    'negative_predictive_value': (metric_mat['tn'], metric_mat['tn'] + metric_mat['fn']),
    'fall_out': (metric_mat['fp'], metric_mat['n']),
    'false_discovery_rate': (metric_mat['fp'], metric_mat['fp'] + metric_mat['tp']),
    'miss_rate': (metric_mat['fn'], metric_mat['fn'] + metric_mat['tp']),
    'accuracy': (metric_mat['tp'] + metric_mat['tn'], metric_mat['total']),
    'f1_score': (2 * metric_mat['tp'], 2 * metric_mat['tp'] + metric_mat['fp'] + metric_mat['fn'])
}

alternative_rate_metric_names = {
    'recall': ['sensitivity', 'true_positive_rate', 'TPR', 'hit_rate'],
    'specificity': ['SPC', 'true_negative_rate'],
    'precision': ['positive_predictive_value', 'PPV'],
    'negative_predictive_value': ['NPV'],
    'fall_out': ['false_positive_rate'],
    'false_discovery_rate': ['FDR'],
    'miss_rate': ['false_negative_rate', 'FNR'],
    'accuracy': ['ACC'],
    'f1_score': ['f1']
}
for root_name, alternatives in alternative_rate_metric_names.iteritems():
    for alt in alternatives:
        rate_metric_mats.update({alt: rate_metric_mats[root_name]})


class SingleMetricGauger(object):
    def __init__(self, actual, probs, prob_thresh=100, metric=None,
                 percentile_thresh=False, **kwargs):
        self.metric_mat = metric_mat
        self.metric_name = kwargs.get('metric_name', None)
        self.prob_name = kwargs.get('prob_name', 'probability')
        self.rate_metric_mats = rate_metric_mats
        self.actual = actual
        self.probs = probs
        self.prob_thresh = prob_thresh
        if isinstance(prob_thresh, int):
            self.prob_thresh = linspace(start=0, stop=1, num=prob_thresh)
        if percentile_thresh:  # interpret prob_thresh as percentiles of y_preds
            self.prob_thresh = percentile(self.probs, list(100 * self.prob_thresh))
        self.prob_thresh = array(self.prob_thresh)
        if isinstance(metric, basestring):
            self.metric_name = self.metric_name or metric
            self.metric = rate_metric_mats[metric]
        else:
            self.metric = metric
        if kwargs.get('compute_rate_metric', None) is not None:
            self.compute_rate_metric = kwargs.get('compute_rate_metric')
        else:
            if isinstance(self.metric, tuple) and len(self.metric) == 2:
                self.compute_metric = compute_rate_metric
            elif shape(self.metric) == (2, 2):
                # consider this as confusion_matrix weights, to be dotted with the confusion matrix and summed
                self.compute_metric = dot_and_sum
        self.last_gauge = None

    @staticmethod
    def mk_single_metric_gauger_with_mean_pred(actual, **kwargs):
        return SingleMetricGauger(actual, probs=mean(actual) * ones((shape(actual))), **kwargs)

    @staticmethod
    def mk_profit_gauger(actual, probs, cost_of_trial, revenue_of_success, **kwargs):
        """
        This gauger emulates the situation where we bet on all items above the probability threshold, incurring a cost
        of cost_of_trial for every such item, and gaining revenue_of_success for every item that succeeds.
        That is, the gauge is the profit:
            tp * revenue_of_success - (fp + tp) * cost_of_trial
        """
        cost_of_trial = abs(cost_of_trial)
        kwargs = dict({'metric_name': 'profit'}, **kwargs)
        return SingleMetricGauger(actual, probs,
                                  metric=array([[0, -cost_of_trial], [0, revenue_of_success - cost_of_trial]]),
                                  compute_metric=dot_and_sum, **kwargs)

    def compute_metric_for_thresh(self, thresh, metric):
        return self.compute_metric(self.confusion_matrix_for_thresh(thresh), metric)

    def confusion_matrix_for_thresh(self, thresh):
        return confusion_matrix(
            y_true=self.actual,
            y_pred=binary_prediction_from_probs_and_thresh(self.probs, thresh))

    def gauge(self, metric=None, prob_thresh=None):
        if metric is not None:
            self.metric = metric
        if prob_thresh is not None:
            self.prob_thresh = prob_thresh
        self.last_gauge = [self.compute_metric_for_thresh(thresh, self.metric)
                           for thresh in self.prob_thresh]
        return self.last_gauge

    def get_gauge(self, metric=None, prob_thresh=None, recompute=False):
        if recompute or self.last_gauge is None:
            self.gauge(metric=metric, prob_thresh=prob_thresh)
        return self.last_gauge

    def plot(self, *args, **kwargs):
        plt.plot(self.prob_thresh, self.last_gauge, *args, **kwargs)
        plt.xlabel(self.prob_name + ' threshold')
        plt.ylabel(self.metric_name)

    def set_metric(self, metric):
        if isinstance(metric, basestring):
            self.metric = rate_metric_mats[metric]
        else:
            self.metric = metric


class MultipleMetricGaugers():
    def __init__(self, actual, probs, prob_thresh=100, metrics=['precision', 'recall'],
                 percentile_thresh=False, **kwargs):
        self.gauger = list()
        for m in metrics:
            self.gauger.append(SingleMetricGauger(actual, probs, prob_thresh, metric=m,
                                                  percentile_thresh=percentile_thresh, **kwargs))

    def plot_metric_against_another(self, i=0, j=1, *args, **kwargs):
        plt.plot(self.gauger[i].get_gauge(), self.gauger[j].get_gauge(), *args, **kwargs)
        if self.gauger[i].metric_name:
            plt.xlabel(self.gauger[i].metric_name)
        if self.gauger[j].metric_name:
            plt.ylabel(self.gauger[j].metric_name)
        plt.grid()
        return plt.gca()


def dot_and_sum(cm, metric):
    return sum(metric * cm)


def compute_rate_metric(cm, metric):
    return sum(metric[0] * cm) / float(sum(metric[1] * cm))


def binary_prediction_from_probs_and_thresh(probas, thresh):
    return array(probas >= thresh, dtype=float)


def confusion_matrix(y_true, y_pred, labels=None):
    cm = sk.metrics.confusion_matrix(y_true, y_pred, labels)
    if shape(cm) == (1, 1):  # bug in sk.metrics.confusion_matrix with all true or all false inputs
        if all(y_true):
            return array([[0, 0], [0, len(y_true)]])
        elif not any(y_true):
            return array([[len(y_true), 0], [0, 0]])
    else:
        return cm


def sensitivity(cm):
    """
    sensitivity or true positive rate (TPR), hit rate, recall
    TPR = TP / P = TP / (TP+FN)
    """
    return cm[1][1] / float(cm[1][1] + cm[1][0])


def recall(cm):
    """
    recall or sensitivity or true positive rate (TPR), hit rate
    TPR = TP / P = TP / (TP+FN)
    """
    return cm[1][1] / float(cm[1][0] + cm[1][1])


def specificity(cm):
    """
    specificity (SPC) or True Negative Rate
    SPC = TN / N = TN / (FP + TN)
    """
    return cm[0][0] / float(cm[0][0] + cm[0][1])


def precision(cm):
    """
    precision or positive predictive value (PPV)
    PPV = TP / (TP + FP)
    """
    t = cm[1][1] / float(cm[1][1] + cm[0][1])
    if isnan(t):
        return 0.0
    else:
        return t

def negative_predictive_value(cm):
    """
    negative predictive value (NPV)
    NPV = TN / (TN + FN)
    """
    return cm[0][0] / float(cm[0][0] + cm[1][0])


def fall_out(cm):
    """
    fall-out or false positive rate (FPR)
    FPR = FP / N = FP / (FP + TN)
    """
    return cm[0][1] / float(cm[0][0] + cm[0][1])


def false_discovery_rate(cm):
    """
    false discovery rate (FDR)
    FDR = FP / (FP + TP) = 1 - PPV
    """
    return cm[0][1] / float(cm[0][1] + cm[1][1])


def miss_rate(cm):
    """
    Miss Rate or False Negative Rate (FNR)
    FNR = FN / (FN + TP)
    """
    return cm[1][0] / float(cm[1][0] + cm[1][1])


def accuracy(cm):
    """
    accuracy (ACC)
    ACC = (TP + TN) / (P + N)
    """
    return (cm[1][1] + cm[0][0]) / float(sum(cm))


def f1_score(cm):
    """
    F1 score is the harmonic mean of precision and sensitivity
    F1 = 2 TP / (2 TP + FP + FN)
    """
    return 2 * cm[1][1] / float(2 * cm[1][1] + cm[0][1] + cm[1][0])


def matthews_correlation_coefficient(cm):
    """
    Matthews correlation coefficient (MCC)
     \frac{ TP \times TN - FP \times FN  {\sqrt{ (TP+FP) ( TP + FN ) ( TN + FP ) ( TN + FN )
    """
    return (cm[1][1] * cm[0][0] - cm[0][1] * cm[1][0]) \
           / sqrt((cm[1][1] + cm[0][1])(cm[1][1] + cm[1][0])(cm[0][0] + cm[0][1])(cm[0][0] + cm[1][0]))


def informedness(cm):
    return sensitivity(cm) + specificity(cm) - 1


def markedness(cm):
    return precision(cm) + negative_predictive_value(cm) - 1


def tn(cm):
    return cm[0][0]


def tp(cm):
    return cm[1][1]


def fp(cm):
    return cm[0][1]


def fn(cm):
    return cm[1][0]