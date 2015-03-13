__author__ = 'thorwhalen'


import numpy as np

def idf_log10(num_of_docs_containing_term, num_of_docs):
    return np.log10(float(num_of_docs) / np.array(num_of_docs_containing_term))
