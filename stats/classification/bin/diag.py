__author__ = 'thor'

from numpy import *
import sklearn as sk


class ZeroEstimator(sk.base.BaseEstimator):
    def __init__(self):
        self._mean_y = nan

    def fit(self, x, y):
        self._mean_y = mean(y)

    def predict(self, x):
        return repeat(0, len(x))

    def predict_proba(self, x):
        return tile([1 - self._mean_y, self._mean_y], [len(x), 1])


class ModelDiagnosis01(object):
    def __init__(self, x, y, test_size=0.5, random_state=None, bench_model=ZeroEstimator):
        self.x = x
        self.y = y  #sk.utils.validation.column_or_1d(y)
        self.test_size = test_size
        self.random_state = random_state
        self.train_x, self.test_x, self.train_y, self.test_y = \
            sk.cross_validation.train_test_split(self.x, self.y,
                                                 test_size=self.test_size,
                                                 random_state=self.random_state)
        self.classification_metrics = {
            'confusion_matrix': sk.metrics.confusion_matrix,
            'accuracy_score': sk.metrics.accuracy_score}
        self.proba_metrics = {
            'log_loss': sk.metrics.log_loss}
        # bench model stuff
        self.bench_model = bench_model()
        self.bench_model.fit(self.train_x, self.train_y)
        self.bench_predicted_y = self.bench_model.predict(self.test_x)
        self.bench_predicted_proba = self.bench_model.predict_proba(self.test_x)
        self.bench_metrics = self.compute_metrics(self.bench_model)[0]

    def compute_metrics(self, model):
        metrics = dict()
        pred_y = model.predict(self.test_x)
        for metric_name, metric_function in self.classification_metrics.iteritems():
            metrics[metric_name] = metric_function(self.test_y, pred_y)
        try:
            pred_proba_y = model.predict_proba(self.test_x)
            for metric_name, metric_function in self.proba_metrics.iteritems():
                metrics[metric_name] = metric_function(self.test_y, pred_proba_y)
        except NotImplementedError:
            pred_proba_y = None
        return metrics, pred_y, pred_proba_y

    def train(self, model):
        model.fit(self.train_x, self.train_y)

    def print_comparison_triple(self, metrics, metric):
        print "%s:  %.04f (%.2f%% of bench (%.04f))" % \
            (metric,
             metrics[metric],
             100. * metrics[metric] / self.bench_metrics[metric],
             self.bench_metrics[metric])

    def test(self, model):
        metrics, pred_y, pred_proba_Y = self.compute_metrics(model)
        print "confusion_matrix:\n%s " % metrics['confusion_matrix']
        for metric_name, metric_value in metrics.iteritems():
            try:
                print "%s:  %.04f (%.2f%% of bench (%.04f))" % \
                    (metric_name,
                     metric_value,
                     100. * metric_value / self.bench_metrics[metric_name],
                     self.bench_metrics[metric_name])
            except TypeError:
                continue
        print sk.metrics.classification_report(self.test_y, pred_y)

    def train_and_test(self, model):
        self.train(model=model)
        self.test(model=model)

    def _process_prob_thresh(self, model, prob_thresh):
        pred_probs = model.predict_proba(self.test_x)[:, 1]
        prob_thresh = prob_thresh or 11
        if isinstance(prob_thresh, int):
            prob_thresh = percentile(pred_probs, linspace(0, 100, prob_thresh))
        return pred_probs, prob_thresh

    def multiple_predict(self, model, prob_thresh=None):
        pred_probs, prob_thresh = self._process_prob_thresh(model, prob_thresh)
        return map(lambda p: (pred_probs > p).astype(float), prob_thresh)

    def multiple_confusion_matrices(self, model, prob_thresh=linspace(0, 1, 0.1)):
        pred_probs, prob_thresh = self._process_prob_thresh(model, prob_thresh)
        return map(lambda p: sk.metrics.confusion_matrix((pred_probs > p).astype(float)), prob_thresh)



