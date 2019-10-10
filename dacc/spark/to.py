

__author__ = 'thor'

import numpy as np
import re


def np_array(rdd):
    return np.array([x for x in rdd.toLocalIterator()])


def sklearn_model(model):
    model_name = re.search("<class '(.+)'", str(model.__class__)).group(1)

    if model_name == 'pyspark.mllib.clustering.GaussianMixtureModel':
        from sklearn.mixture import GMM
        n_components = len(model.gaussians)
        sk_model = GMM(n_components=n_components, covariance_type='full')
        sk_model.weights_ = model.weights
        sk_model.means_ = np.array([x.mu for x in model.gaussians])
        sk_model.covars_ = np.array([x.sigma.toArray() for x in model.gaussians])
    elif model_name == 'pyspark.mllib.classification.NaiveBayesModel':
        pass

    return sk_model