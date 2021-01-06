"""Multi classifier testing"""
__author__ = 'thor'

import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn import ensemble
from sklearn.feature_extraction import text
from sklearn import feature_extraction
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import linear_model
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import svm
from sklearn import tree

from ut.util.log import printProgress
from ut.daf.manip import reorder_columns_as

default_scorers = {
    'accuracy': metrics.accuracy_score
}

default_score_aggreg = [
    np.mean,
    np.min
]

default_classifiers = [
    svm.LinearSVC(random_state=0),
    svm.SVC(random_state=0),
    linear_model.LogisticRegression(),
    tree.DecisionTreeClassifier(),
    naive_bayes.BernoulliNB(),
    naive_bayes.MultinomialNB(),
    naive_bayes.GaussianNB(),
    linear_model.SGDClassifier(),
    linear_model.RidgeClassifier(),
    ensemble.RandomForestClassifier(n_estimators=10)
]

plt.style.use('fivethirtyeight')

_PLT_LEGEND_OPTIONS = dict(loc="upper center",
                           bbox_to_anchor=(0.5, -0.15),
                           fancybox=True,
                           shadow=True,
                           ncol=3)

colors = [ii.strip() for ii in '#30a2da, #fc4f30, #e5ae38, #6d904f, #8b8b8b'.split(',')]
colors += ['#' + ii.strip() for ii in
           '348ABD, A60628, 7A68A6, 467821,D55E00,  CC79A7, 56B4E9, 009E73, F0E442, 0072B2'.split(',')]
markers = itertools.cycle(["o", "D"])
colors = itertools.cycle(colors)


def score_classifier(X, y, clf, nfeats=None,
                     scoring=default_scorers, score_aggreg=default_score_aggreg,
                     scale=None, decompose=None, select=None, decompose_params={},
                     nfolds=10, shuffle=True, random_fold_state=None,
                     include_train_stats=False):
    """
    Tests the CLF classifier with NFOLDS of train/test splits, and scores the results using one or several score
    functions (specified by SCORING), returning a pandas Series listing the aggregate(s) (specified by SCORE_AGGREG)
    of the scores.
    """
    # give scoring and score_aggreg elements some names
    scoring = scoring or default_scorers
    scoring = mk_scoring_dict(scoring)
    score_aggreg = score_aggreg or default_score_aggreg
    score_aggreg = mk_score_aggreg_dict(score_aggreg)

    if nfeats is None:
        nfeats = np.shape(X)[1]

    # X = X[:, :nfeats]

    stratified_k_fold = StratifiedKFold(y, n_folds=nfolds,
                                        shuffle=shuffle,
                                        random_state=random_fold_state)
    score_info = list()
    for train, test in stratified_k_fold:
        d = dict()

        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        if include_train_stats:
            d['train_pts'] = np.shape(X_train)[0]
            d['train_nfeats'] = np.shape(X_train)[1]

        pipeline_steps = list()
        if scale:  # preprocessing.StandardScaler(), preprocessing.MinMaxScaler()
            pipeline_steps.append(('scale', scale))
        if decompose:
            pipeline_steps.append(('decompose', decompose))
        if select:
            pipeline_steps.append(('select', feature_selection.SelectKBest(k=nfeats)))
        else:
            X = X[:, :nfeats]

        pipeline_steps.append(('clf', clf))

        pipeline = Pipeline(steps=pipeline_steps)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        for score_name, score_fun in scoring.items():
            d[score_name] = score_fun(y_test, y_pred)
        score_info.append(d)

    # return score_info
    score_info = pd.DataFrame(score_info)
    score_result = pd.Series()
    for score_aggreg_name, score_aggreg_fun in score_aggreg.items():
        t = score_info.apply(score_aggreg_fun)
        t.set_axis(axis=0,
                   labels=[mk_aggreg_score_name(score_aggreg_name, score_name) for score_name in t.index.values])
        score_result = score_result.append(t)

    return score_result


def test_classifiers(X, y,
                     scoring=default_scorers,
                     score_aggreg=default_score_aggreg,
                     n_features=7,
                     # an int will be transformed to a list (with different num of features) of given size
                     clfs=None,
                     nfolds=10,
                     scale=None,
                     decompose=None,
                     select=None,
                     decompose_params={},
                     print_progress=False,
                     score_to_plot=None
                     ):
    """
    tests and scores (given by SCORING and SCORE_AGGREG) several classifiers (given by clfs) with several number of
    features, returning a pandas DataFrame of the results.
    """
    scoring = scoring or default_scorers
    score_aggreg = score_aggreg or default_score_aggreg

    if isinstance(n_features, int):  # if n_features is an int, it's the number of different feature set lens to try out
        # ... so make this feature set len list
        total_n_features = np.shape(X)[1]
        n_features = list(range(1, total_n_features + 1, int(np.floor(total_n_features / n_features))))[:n_features]
    y = np.asarray(y, dtype="|S6")
    n_features = np.array(n_features)

    if clfs is None:
        clfs = default_classifiers

    clfs = clfs_to_dict_clfs(clfs)

    general_info_dict = dict()
    if scale is not None and scale is not False:  # preprocessing.StandardScaler(), preprocessing.MinMaxScaler()
        if scale is True:
            scale = preprocessing.StandardScaler()
        general_info_dict['scale'] = get_name(scale)
    if decompose is not None and decompose is not False:
        if decompose is True:
            decompose = decomposition.PCA(
                **decompose_params)  # PCA, KernelPCA, ProbabilisticPCA, RandomizedPCA, TruncatedSVD
        general_info_dict['decompose'] = get_name(decompose)

    clf_results = list()

    for i_nfeats, nfeats in enumerate(n_features):
        for i_clf, clf in enumerate(clfs):
            clf_name = list(clf.keys())[0]
            clf = clf[clf_name]
            d = dict(general_info_dict, **{'model': clf_name, 'nfeats': nfeats})
            if print_progress:
                printProgress("{}: nfeats={}, nfolds={}".format(
                    clf_name,
                    n_features[i_nfeats],
                    nfolds))
            # try:
            start_time = datetime.now()
            score_result = \
                score_classifier(X,
                                 y,
                                 clf=clf,
                                 nfeats=nfeats,
                                 scoring=scoring,
                                 score_aggreg=score_aggreg,
                                 nfolds=nfolds,
                                 scale=scale,
                                 decompose=decompose,
                                 select=select,
                                 decompose_params=decompose_params)
            d.update({'seconds': (datetime.now() - start_time).total_seconds()})
            d.update(score_result.to_dict())
            # except ValueError as e:
            #     raise e
            #     print("Error with: {} ({} features)".format(get_name(clf),
            #                                         n_features[i_nfeats]))

            clf_results.append(d)  # accumulate results

    clf_results = pd.DataFrame(clf_results)
    if score_to_plot:
        if score_to_plot is True:
            score_to_plot = mk_aggreg_score_name(score_aggreg_name=list(mk_score_aggreg_dict(score_aggreg).keys())[0],
                                                 score_name=list(mk_scoring_dict(scoring).keys())[0])
        plot_score(clf_results, score_to_plot)

    return reorder_columns_as(clf_results, ['model', 'nfeats', 'seconds'])


def clfs_to_dict_clfs(clfs):
    for i, clf in enumerate(clfs):
        if not isinstance(clf, dict):
            clfs[i] = {get_name(clf): clf}
    return clfs


def decompose_data(X, decompose, n_components=None, y=None, decompose_params={}):
    if n_components is None:
        n_components = np.shape(X)[1]
    try:
        decomposer = decompose(n_components=n_components, whiten=True, **decompose_params)
    except TypeError:
        print(("No whiten option in {}".format(decompose)))
        decomposer = decompose(n_components=n_components, **decompose_params)
    try:
        if y is None:
            decomposer.fit(X)
        else:
            decomposer.fit(X, y)
    except ValueError:
        decomposer = decompose(n_components=n_components - 1, **decompose_params)
        if y is None:
            decomposer.fit(X)
        else:
            decomposer.fit(X, y)
    return decomposer.transform(X)


def plot_score(clf_results, score_to_plot, parameter='nfeats', **kwargs):
    # defaults
    kwargs = dict(dict(figsize=(7, 5)), **kwargs)

    t = clf_results[['model', parameter, score_to_plot]] \
        .set_index(['model', parameter]).unstack('model')[score_to_plot]

    ax = t.plot(**kwargs)
    plt.xlabel(parameter)
    plt.ylabel(score_to_plot)
    plt.title("{} vs {}".format(score_to_plot, parameter))
    return ax


def get_name(obj):
    if hasattr(obj, '__name__'):
        return obj.__name__
    else:
        return type(obj).__name__


def mk_scoring_dict(scoring):
    if not isinstance(scoring, dict):
        if not hasattr(scoring, '__iter__'):
            scoring = [scoring]
        scoring = {x.__name__: x for x in scoring}
    return scoring


def mk_score_aggreg_dict(score_aggreg):
    if not isinstance(score_aggreg, dict):
        if not hasattr(score_aggreg, '__iter__'):
            score_aggreg = {'': score_aggreg}
        else:
            score_aggreg = {x.__name__: x for x in score_aggreg}
    return score_aggreg


def mk_aggreg_score_name(score_aggreg_name, score_name):
    if score_aggreg_name:
        return score_aggreg_name + '_' + score_name
    else:
        return score_name


def __main__():
    from sklearn.svm import LinearSVC
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    clf_results = test_classifiers(X, y,
                                   scoring=metrics.accuracy_score,
                                   n_features=list(range(1, np.shape(X)[1])),
                                   clfs=None,
                                   print_progress=False,
                                   score_to_plot=None)

    print(clf_results)
