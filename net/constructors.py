

import itertools
from scipy import sparse


def pair_count_sr_to_coo_matrix(sr):
    """ Takes a (i,j)->count series and makes a sparse (weighted) adjacency matrix (coo_matrix) out of it """
    data, i, j = list(zip(*[(x[1], x[0][0], x[0][1]) for x in iter(sr.items())]))
    return sparse.coo_matrix((data, (i, j)))




