from __future__ import division

__author__ = 'thor'

import ut.dacc.spark.util as sutil
import ut.dacc.spark.to as sto


class RDD(object):
    def __init__(self, rdd):
        self.rdd = rdd

    def head(self, k=5):
        return sutil.head(self.rdd, k)

    def array(self):
        return sto.np_array(self.rdd)

    def iterate(self):
        return self.rdd.toLocalIterator()


