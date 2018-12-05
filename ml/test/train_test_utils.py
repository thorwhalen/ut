from __future__ import division

from collections import Counter
from inspect import isclass

from numpy import isnan

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def name_obj_from_obj(obj):
    return (obj.__class__.__name__, obj)


scores_for = {}

scores_for['reg'] = [
    'explained_variance_score', 'mean_absolute_error', 'mean_squared_error',
    'median_absolute_error', 'r2_score']

scores_for['clf'] = [
    'accuracy_score', 'average_precision_score', 'brier_score_loss',
    'precision_score', 'recall_score', 'roc_auc_score']


def model_scores(y_true, y_pred, model_type='clf'):
    s = dict()
    if model_type is not None:
        for score_func in scores_for[model_type]:
            try:
                s[score_func] = getattr(metrics, score_func)(y_true, y_pred)
            except Exception:
                pass
    else:
        for score_func in scores_for['reg'] + scores_for['clf']:
            try:
                s[score_func] = getattr(metrics, score_func)(y_true, y_pred)
            except Exception:
                pass

    return s


def safe_element_isnan(x):
    return x == 'nan' or not isinstance(x, basestring) and isnan(x)


def type_of_elements_of_arr(arr, max_count_for_categoricals=9):
    """
    Returns the (statistical) type of data in the arr.
    The function infers this type by removing nans (numpy nan or 'nan' string),
    then looking at the most common element
    :param arr:
    :param max_count_for_categoricals:
    :return:

    >>> from numpy import nan
    >>> print(safe_isnan(['asd', 3, nan, 'nan']))
    [False False  True  True]
    >>> print(type_of_elements_of_arr(['foo', 'bar', 1, nan, nan, nan]))
    categorical
    >>> print(type_of_elements_of_arr([1, 2, 'foo', nan, nan, 'nan', 'nan', 'nan']))
    categorical
    >>> print(type_of_elements_of_arr(range(20) + ['foo', nan, nan, 'nan', 'nan', 'nan']))
    numerical
    >>> print(type_of_elements_of_arr(map(str, range(100))))
    nominal
    """
    unnanned_arr = [x for x in arr if not safe_element_isnan(x)]
    n = len(unnanned_arr)
    if n == 0:
        return 'empty'
    counts = Counter(unnanned_arr)
    if len(counts) <= max_count_for_categoricals:
        return 'categorical'
    else:
        py_type = Counter(type(x) for x in arr if not safe_element_isnan(x)).most_common(1)[0][0]
        if py_type == str:
            return 'nominal'
        else:
            return 'numerical'


def default_model_type_for_y(y):
    y_type = type_of_elements_of_arr(y)
    if y_type == 'numerical':
        return 'reg'
    else:
        return 'clf'


def multiple_train_test_stats(model,
                              X,
                              y,
                              test_sizes=(0.25, 0.5),
                              n_tests_per_size=1,
                              model_type=None):
    """
    Perform multiple train/test splits on data, train models, test them,
    and gather statistics about these tests.
    :param model: The model object, class, or function (without arguments) to create one
    :param X: The explanatory variables matrix
    :param y: The target variable
    :param test_sizes: The test sizes (or proportions) to use during train/test splits
    :param n_tests_per_size: The number of splits to perform per size
    :param model_type: A specification of what type of model this is ('reg' or 'clf'). Will try
    to figure it out from the y array if not specified.
    :return:
    """

    if isclass(model):
        get_model = lambda: model()
    else:
        get_model = lambda: model

    model_type = model_type or default_model_type_for_y(y)

    if isinstance(test_sizes, (float, int)):
        test_sizes = [test_sizes]
    stats = list()

    for test_size in test_sizes:
        for _ in xrange(n_tests_per_size):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            d = {'n_train_pts': len(y_train), 'n_test_pts': len(y_test)}

            model = get_model().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            d = dict(d, **model_scores(y_true=y_test, y_pred=y_pred, model_type=model_type))

            stats.append(d)

    return stats


dflt_regressor = LinearRegression
dflt_classifier = SVC
model_type_for_y_type = {'numerical': 'reg', 'categorical': 'clf'}
dflt_model_for_model_type = {'reg': dflt_regressor, 'clf': dflt_classifier}


def test_datasets(list_of_name_and_dataset_pairs,
                  model_for_model_type=dflt_model_for_model_type,
                  test_size=0.25, random_state=42
                  ):
    raise NotImplementedError("Not finished implementing yet.")
    stats = list()
    problematic_pairs = list()

    for dataset_name, dataset in list_of_name_and_dataset_pairs:
        for y_var, y_type in dataset.type_of_target.iteritems():
            d = {'dataset': dataset_name, 'y_var': y_var, 'y_type': y_type}
            try:
                X, y = dataset.get_x_and_y(y=y_var)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state)
                d['n_train_pts'] = len(y_train)
                d['n_test_pts'] = len(y_test)

                model_type = model_type_for_y_type[y_type]
                model = model_for_model_type[model_type]()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                d = dict(d, **model_scores(y_true=y_test, y_pred=y_pred, model_type=model_type))

                stats.append(d)

            except Exception:
                try:
                    prob_pair = (str(dataset.name), str(y_var))
                    print('could not deal with that one: {}'.format(prob_pair))
                    problematic_pairs.append(prob_pair)
                except Exception as e:
                    print(e)
                continue

    return stats
