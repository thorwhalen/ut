from collections import defaultdict
from oto.models.scent import CentroidSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dflt_metrics = (
    ('accuracy', accuracy_score),
    ('f1', f1_score)
)


def gather_multiple_train_test_results(X, y,
                                       model,
                                       test_sizes=tuple(np.linspace(.1, .9, 9)),
                                       n_tests_per_test_size=100,
                                       metrics=dflt_metrics
                                       ):
    r = defaultdict(lambda: defaultdict(list))
    for test_size in test_sizes:
        for _ in range(n_tests_per_test_size):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            for metric_name, metric_func in metrics:
                r[metric_name][test_size].append(metric_func(y_true=y_test, y_pred=y_pred))

    return r
