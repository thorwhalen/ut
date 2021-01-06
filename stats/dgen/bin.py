"""Generating binary data"""
__author__ = 'thor'

from numpy import *
import pandas as pd


def binomial_mixture(npts=10,
                     success_prob=None,
                     mixture=None,
                     n_components=2,
                     n_trials=[1, 20],
                     include_component_idx=False,
                     include_component_prob=False,
                     **kwargs):
    if success_prob is not None:
        n_components = len(success_prob)
    elif mixture is not None:
        n_components = len(mixture)

    if success_prob is None:
        success_prob = random.rand(n_components)
    success_prob = array(success_prob)
    if mixture is None:
        mixture = random.rand(n_components)
    mixture = array(mixture)
    mixture = mixture / sum(mixture)

    n_trials_col = kwargs.get('n_trials_col', 'n_trials')
    n_success_col = kwargs.get('n_success_col', 'n_success')

    if callable(n_trials):
        n_trials = n_trials(npts)
    else:
        if isinstance(n_trials, int):
            n_trials = [1, n_trials]
        if len(n_trials) == 2:  # assume min and max trials are given
            n_trials = random.random_integers(n_trials[0], n_trials[1], npts)

    data = pd.DataFrame({n_trials_col: n_trials})
    component_idx = array(mixture).cumsum().searchsorted(random.sample(npts))
    component_prob = success_prob[component_idx]
    data[n_success_col] = random.binomial(n_trials, component_prob)

    if include_component_idx:
        data['component_idx'] = component_idx
    if include_component_prob:
        data['component_prob'] = component_prob

    return data

