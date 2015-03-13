__author__ = 'thor'


from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def bell(actual, probs, failure_color='red', success_color='blue',
         failure_weight=1, success_weight=1,
         log_scale=False):
    """
    Plots probs separating failures (actual=False) and successes (actual=True), in the following way:
        * left (x < 0) side, the probs for failures are plotted, increasingly
        * right (x > 0) side, the probs for successes are plotted, decreasingly

    Options:
        failure_weight=None will result in weights being set so that there's a balanced num * weight for
        for failure and success (this is to be able to compare the shapes of the curves more easily)
    """

    t = pd.DataFrame({'actual': actual, 'probs': probs})
    success_lidx = array(t['actual'])
    n_success = sum(success_lidx)
    failure_lidx = ~success_lidx
    n_failure = sum(failure_lidx)

    if failure_weight is None:
        failure_weight = 1. / n_failure
        success_weight = 1. / n_success

    failure_d = t[failure_lidx].sort('probs', ascending=True)
    failure_d['x'] = arange(- failure_weight * n_failure, 0, failure_weight)
    success_d = t[success_lidx].sort('probs', ascending=False)
    success_d['x'] = arange(0, success_weight * n_success, success_weight) + success_weight

    plt.plot(failure_d['x'], failure_d['probs'], failure_color,
             success_d['x'], success_d['probs'], success_color)

    ax = plt.gca()

    # ax.fill_between(failure_d['x'], 0, failure_d['probs'], color=failure_color)
    # ax.fill_between(success_d['x'], 0, success_d['probs'], color=success_color)

    if log_scale:
        print "Yeah, there's still a bug here. Wanna repair it?"
        ax.set_yscale('log', basex=2)
        min_val = min(t['probs'][t['probs'] > 0])
    else:
        min_val = 0

    ax.fill_between(failure_d['x'], min_val, failure_d['probs'], color=failure_color)
    ax.fill_between(success_d['x'], min_val, success_d['probs'], color=success_color)
    plt.xlim(failure_d['x'].iloc[0], success_d['x'].iloc[-1])
    plt.ylim(min_val, max(t['probs']))
    return ax