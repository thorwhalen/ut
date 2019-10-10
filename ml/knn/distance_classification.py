__author__ = 'thor'

import numpy as np
from numpy import *


def multiple_similarity_alignment(simil_matrix, y):
    """
    Let $x, y, z$ be items such that $y,z\in C$ but $x\notin C$.
    A similarity $s$ function is good on this pair if $y$ is is more similar to $z$ then $x$ is, i.e. $s(x,z) \leq s(y,z)$.

    If with have $n$ items, $k$ of which are in $C$,
    then there's $k(n-k)$ possible pairs that could be checked for this alignment.

    We will gauge the alignment of the similarity $s$ with respect to the classification $C$ as the proportion of good
    (i.e. aligned) pairs to the total number of pairs that could be misclassified.

    """
    probability_of_alignment = dict()
    number_of_aligned_pairs = dict()
    total_num_of_error_possibilities = dict()
    for label in unique(y):
        class_idx = where(y == label)[0]
        probability_of_alignment[label], number_of_aligned_pairs[label], total_num_of_error_possibilities[label] = \
            similarity_alignment(
                mk_simil_ordered_same_class_pairs_bitmap(
                    simil_matrix, class_idx
                )
            )
    aggregate_prob_of_alignment = \
        np.sum(list(number_of_aligned_pairs.values())) / np.sum(list(total_num_of_error_possibilities.values()))

    return aggregate_prob_of_alignment, \
           probability_of_alignment, \
           number_of_aligned_pairs, \
           total_num_of_error_possibilities


def similarity_alignment(simil_ordered_same_class_pairs_bitmap):
    # start with a bit map indexed by all possible 2-sets of items, ordered in increasing order of similarity,
    # and indicating with a 1/True when the items of the 2-set are of the same target class C,
    # and by 0/False if one is from C and the other not.
    simil_ordered_same_class_pairs_bitmap = array(list(map(bool, simil_ordered_same_class_pairs_bitmap)))

    # The total number of error possibilities: The number of (0,1) or (1,0) pairs in
    total_num_of_error_possibilities = \
        float(sum(simil_ordered_same_class_pairs_bitmap) * sum(~simil_ordered_same_class_pairs_bitmap))

    # Cumul of the number of "both in class" pairs when traversing the sequence
    cum_t = cumsum(simil_ordered_same_class_pairs_bitmap)

    # The position of the "both in class" pairs
    out_of_class_idx = where(~simil_ordered_same_class_pairs_bitmap)[0]


    # An array showing for every out_of_class_idx, how many in class elements were before.
    cum_t[out_of_class_idx]
    # print(cum_t[out_of_class_idx]); print("")

    # Taking total_num_of_error_possibilities minus the sum of the number of pairs of (same, different)
    # pairs that such that simil(same) < simil(different)
    number_of_aligned_pairs = total_num_of_error_possibilities - np.sum(cum_t[out_of_class_idx])
    # One minus that number is the probability that a randomly chosen (in, out) or (out, in) has similarities aligned
    # with classification
    number_of_aligned_pairs = total_num_of_error_possibilities - np.sum(cum_t[out_of_class_idx])
    probability_of_alignment = number_of_aligned_pairs / total_num_of_error_possibilities
    return probability_of_alignment, number_of_aligned_pairs, total_num_of_error_possibilities


def _class_band(simil_matrix, class_idx):
    return simil_matrix[class_idx, :]


def mk_simil_ordered_same_class_pairs_bitmap(simil_matrix, class_idx):
    """
    Input:
        simil_matrix: the similarity matrix (must be symetrical)
        class_idx: the indices of the targeted class
    returns two arrays:
        ordered_similarities: similarities of all pairs of distinct items containing at least one item of the class,
            ordered by increasing similarity
        simil_ordered_same_class_pairs_bitmap: aligned with ordered_similarities,
            indicating whether both items of the pair are in the same (target) class
    """
    n = simil_matrix.shape[0]
    k = len(class_idx)
    band = _class_band(simil_matrix, class_idx)
    permi = np.append(class_idx, [x for x in range(n) if x not in class_idx])
    band = band[:, permi]
    simil_ordered_same_class_pairs_bitmap = np.append(np.ones((k, k)), np.zeros((k, n - k)), axis=1).astype(bool)

    mask = np.triu(np.ones(band.shape).astype(bool), k=1)

    ordered_similarities = band[mask]
    permi = np.argsort(ordered_similarities)
    ordered_similarities = ordered_similarities[permi]
    simil_ordered_same_class_pairs_bitmap = simil_ordered_same_class_pairs_bitmap[mask][permi]

    return simil_ordered_same_class_pairs_bitmap


