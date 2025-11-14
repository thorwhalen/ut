__author__ = 'thor'

from numpy import *
import sklearn as sk
from scipy.optimize import minimize_scalar
import pandas as pd
import matplotlib.pyplot as plt


class ProbAndRateAnalysis:
    def __init__(self, actual, probs):
        self.d = pd.DataFrame({'probs': probs, 'actual': actual}, columns=['probs', 'actual'])
        self.d['actual'] = self.d['actual'] > 0
        self.d.apply(random.shuffle, axis=0)
        self.d = self.d.sort(['probs']).reset_index(drop=True)
        self.num_of_points = len(self.d)
        self.num_of_positives = self.d['actual'].sum()
        self.profitable_ratio = float(self.num_of_positives) / self.num_of_points

    def _decision_lidx(self, decision_thresh):
        return array(self.d['probs'] >= decision_thresh)

    def _decision_thresh_with_min_trials(self, min_trials):
        min_func = lambda x: abs(self.trials_and_successes_at_decision_threshold(x)[0] - min_trials)
        res = minimize_scalar(min_func, method='bounded', bounds=(0, 1.0))
        return res['x']

    def trials_and_successes_at_decision_threshold(self, decision_thresh):
        lidx = self._decision_lidx(decision_thresh)
        trials = float(sum(lidx))
        successes = float(sum(self.d['actual'].iloc[lidx]))
        return trials, successes

    def trials_and_successes_at_decision_threshold_df(self, decision_thresh=None):
        if decision_thresh is None:
            decision_thresh = 300  # = min_trials
        if isinstance(decision_thresh, int) and decision_thresh > 1:
            min_trials = self._decision_thresh_with_min_trials(decision_thresh)
            decision_thresh = linspace(0.0, min_trials, 101)

        df = pd.DataFrame()
        df['decision threshold'] = decision_thresh
        df = pd.concat([df,
                        pd.DataFrame(list(map(self.trials_and_successes_at_decision_threshold, decision_thresh)),
                                     columns=['trials', 'successes'])],
                       axis=1)
        df['rate'] = df['successes'] / df['trials']
        return df

    def plot_thresh_and_true_rate(self, decision_thresh=None):
        t = self.trials_and_successes_at_decision_threshold_df(decision_thresh)
        plt.plot(t['decision threshold'], t['rate'])
        plt.plot(plt.xlim(), plt.xlim(), 'k:')
        plt.xlabel('decision threshold')
        plt.ylabel('rate for data whose prob is above that threshold')
        plt.tight_layout()

    def bootstrap_samples_df(self, n_samples=500, sample_size=None, replace=True):
        if sample_size is None:
            sample_size = int(10 / self.profitable_ratio)
        return pd.DataFrame(data=[self.d.ix[random.choice(self.d.index, sample_size, replace)].mean() for x in range(n_samples)],
                            columns=self.d.columns)

    def plot_bootstrap_scatter(self, n_samples=500, sample_size=None, replace=True, alpha=0.2):
        df = self.bootstrap_samples_df(n_samples=n_samples, sample_size=sample_size, replace=replace)
        df.plot(kind='scatter', x='probs', y='actual', alpha=alpha)
        lim = max([plt.xlim(), plt.ylim()])
        plt.plot(lim, lim, 'k:')
        plt.axis('tight')

    def plot_bootstrap_error_hist(self, bins=None, n_samples=500, sample_size=None, replace=True):
        if bins is None:
            bins = min([100, int(n_samples / 30)])
        df = self.bootstrap_samples_df(n_samples=n_samples, sample_size=sample_size, replace=replace)
        df = df['actual'] - df['probs']
        df.plot(kind='hist', bins=50)

    def prob_and_rate_bins_df(self, min_trials=None):
        if min_trials is None:
            min_trials = int(10 / self.profitable_ratio)
        from sklearn import tree
        clf = tree.DecisionTreeClassifier(min_samples_leaf=500)
        X = reshape(self.d['probs'], (self.num_of_points, 1)).astype(float32)
        y = reshape(self.d['actual'], (self.num_of_points, 1)).astype(float32)
        clf.fit(X, y)
        thresh = unique(clf.tree_.threshold)[1:]
        df = pd.DataFrame(hstack((X, y)), columns=['probs', 'actual'])
        return df.groupby(digitize(X[:, 0], thresh)).mean()

    def plot_prob_and_rate_bins(self, min_trials=None):
        self.prob_and_rate_bins_df(min_trials=min_trials).plot()


class ZeroEstimator(sk.base.BaseEstimator):
    def __init__(self):
        self._mean_y = nan

    def fit(self, x, y):
        self._mean_y = mean(y)

    def predict(self, x):
        return repeat(0, len(x))

    def predict_proba(self, x):
        return tile([1 - self._mean_y, self._mean_y], [len(x), 1])


class ModelDiagnosis01:
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
        for metric_name, metric_function in self.classification_metrics.items():
            metrics[metric_name] = metric_function(self.test_y, pred_y)
        try:
            pred_proba_y = model.predict_proba(self.test_x)
            for metric_name, metric_function in self.proba_metrics.items():
                metrics[metric_name] = metric_function(self.test_y, pred_proba_y)
        except NotImplementedError:
            pred_proba_y = None
        return metrics, pred_y, pred_proba_y

    def train(self, model):
        model.fit(self.train_x, self.train_y)

    def print_comparison_triple(self, metrics, metric):
        print("%s:  %.04f (%.2f%% of bench (%.04f))" % \
            (metric,
             metrics[metric],
             100. * metrics[metric] / self.bench_metrics[metric],
             self.bench_metrics[metric]))

    def test(self, model):
        metrics, pred_y, pred_proba_Y = self.compute_metrics(model)
        print("confusion_matrix:\n%s " % metrics['confusion_matrix'])
        for metric_name, metric_value in metrics.items():
            try:
                print("%s:  %.04f (%.2f%% of bench (%.04f))" % \
                    (metric_name,
                     metric_value,
                     100. * metric_value / self.bench_metrics[metric_name],
                     self.bench_metrics[metric_name]))
            except TypeError:
                continue
        print(sk.metrics.classification_report(self.test_y, pred_y))

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
        return [(pred_probs > p).astype(float) for p in prob_thresh]

    def multiple_confusion_matrices(self, model, prob_thresh=linspace(0, 1, 0.1)):
        pred_probs, prob_thresh = self._process_prob_thresh(model, prob_thresh)
        return [sk.metrics.confusion_matrix((pred_probs > p).astype(float)) for p in prob_thresh]



