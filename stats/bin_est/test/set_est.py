__author__ = 'thor'

from numpy import *
import pandas as pd
from ut.stats.bin_est.set_est import Shapley


def test_shapley():
    """
    This test tests shapley value calculation using the example in http://www.bis.org/publ/qtrpdf/r_qt0909y.htm
    """
    w = pd.DataFrame(
        index=[('A',), ('B',), ('C',), ('A','B'), ('A','C'), ('B','C'), ('A','B','C')],
        data={'success': [4, 4, 4, 9, 10, 11, 15]}
    )
    print(w)
    se = Shapley(d=w)
    assert se.compute_shapley_values() == {'A': 4.5, 'B': 5.0, 'C': 5.5}, "test_shapley FAILED!"