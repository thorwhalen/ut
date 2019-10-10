__author__ = 'thor'


import scipy.stats
from sklearn.preprocessing import normalize
import pandas as pd
from numpy import *
from datetime import datetime, timedelta


class RandomHour(object):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.norm_rand = scipy.stats.norm(loc=loc, scale=scale).rvs

    def rvs(self, size=None):
        return mod(self.norm_rand(size=size), 24)


def get_rand_function(params,
                      rand_class=scipy.stats.norm,
                      default_mean=0.0,
                      default_std=None):

    default_std = default_std or default_mean / 2.0

    if params is None:
        params = dict(loc=default_mean, scale=default_std)
    elif isinstance(params, int) or isinstance(params, float):
        params = dict(loc=params, scale=default_std)
    elif isinstance(params, tuple):
        params = dict(loc=params[0], scale=params[1])

    if isinstance(rand_class, str):
        if rand_class == 'random_hour':
            rand_class = RandomHour
        else:
            raise ValueError("Unkown rand_class")

    if isinstance(params, dict):
        return rand_class(**params).rvs
    elif hasattr(params, '__call__'):
        return params
    else:
        raise TypeError("Unknown params type")


default_hour_of_visit = array([
    0.04130776, 0.03085015, 0.015666, 0.01200057, 0.00792214,
    0.01122844, 0.01364356, 0.02128387, 0.03329782, 0.04320056,
    0.04705777, 0.04778925, 0.05337438, 0.05587053, 0.05888794,
    0.06357238, 0.06693209, 0.06427901, 0.05929982, 0.05592206,
    0.05495461, 0.04827613, 0.04718839, 0.04619477])

HOURS_PER_DAY = 24.
SECONDS_PER_DAY = HOURS_PER_DAY * 60. * 60.


class UserModel(object):
    def __init__(self,
                 anchor_date=None,
                 hour_of_visit=None,
                 inactivity_lag_days=None,
                 visit_pages=None,
                 seconds_on_page=None,
                 _id=None
                ):

        self.anchor_date = anchor_date or datetime(2014, 1, 1)

        hour_of_visit = hour_of_visit or default_hour_of_visit
        if isinstance(hour_of_visit, ndarray) and len(hour_of_visit) == 24:
            hour_of_visit = random.choice(a=list(range(len(hour_of_visit))), p=hour_of_visit)
        self.hour_of_visit = \
            get_rand_function(params={'loc': hour_of_visit, 'scale': 2.0},
                              rand_class=RandomHour)

        inactivity_lag_days = inactivity_lag_days or scipy.stats.poisson.rvs(mu=7, loc=1)
        self.inactivity_lag_days = \
            get_rand_function(params={'mu': inactivity_lag_days},
                              rand_class=scipy.stats.poisson)

        visit_pages = visit_pages or scipy.stats.poisson.rvs(mu=3, loc=1)
        self.visit_pages = \
            get_rand_function(params={'mu': visit_pages, 'loc': 1},
                              rand_class=scipy.stats.poisson)

        seconds_on_page = seconds_on_page or scipy.stats.lognorm.rvs(s=1, loc=25)
        self.seconds_on_page = \
            get_rand_function(params={'s': 1,
                                      'loc': seconds_on_page,
                                      'scale': seconds_on_page / 2.0
                                     },
                              rand_class=scipy.stats.lognorm)

        self._id = _id or random.randint(1e16)

        self.current_date = self.anchor_date

    def next_session_data(self):
        current_date_midnight = datetime.fromordinal(self.current_date.date().toordinal())  # round date to day level
        self.current_date = current_date_midnight \
                            + timedelta(self.inactivity_lag_days() + self.hour_of_visit() / HOURS_PER_DAY)
        current_time = self.current_date
        session_data = list()
        for i in range(self.visit_pages()):
            current_time += timedelta(self.seconds_on_page() / SECONDS_PER_DAY)
            session_data.append({'_id': self._id, 'date': current_time})
        return session_data

    def __getstate__(self): return self.__dict__
    
    def __setstate__(self, d): self.__dict__.update(d)


long_function = {(): 0, (1,): 0, (2,): 0, (3,): 0, (4,): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0, (2, 3): 0, (2, 4): 0,
                 (3, 4): 0, (1, 2, 3): 0, (1, 2, 4): 45, (1, 3, 4): 40, (2, 3, 4): 0, (1, 2, 3, 4): 65}