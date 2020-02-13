inf = float('infinity')


def gen_nearest_neighbor_matching_pairs(query_pts, match_pts, radius=inf, yield_indices_only=False):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    query_pts = np.array(query_pts)
    match_pts = np.array(match_pts)

    def trans_wrt_ndim(arr):
        ndim = np.ndim(arr)
        if ndim == 1:
            return np.reshape(arr, (-1, 1))
        else:
            return arr

    if isinstance(match_pts, NearestNeighbors):
        nn = match_pts
    else:
        nn = NearestNeighbors(n_neighbors=1).fit(trans_wrt_ndim(match_pts))

    match_dist, match_idx = nn.kneighbors(trans_wrt_ndim(query_pts), return_distance=True)
    match_dist = match_dist[:, 0]
    match_idx = match_idx[:, 0]
    within_radius_lidx = match_dist <= radius

    if yield_indices_only:
        yield from zip(np.arange(len(within_radius_lidx))[within_radius_lidx], match_idx[within_radius_lidx])
    else:
        yield from zip(query_pts[within_radius_lidx], match_pts[match_idx[within_radius_lidx]])


def _first_pair_within_radius(it1, x2, radius=inf, dist_func=lambda x1, x2: abs(x1 - x2)):
    for i, x1 in enumerate(it1):
        if x1 > x2:
            return i, None
        elif dist_func(x1, x2) <= radius:
            return i, x1
    return None, None


def _close_pairs_helper(it1, it2, radius=inf, inverted=False, dist_func=lambda x1, x2: abs(x1 - x2)):
    if len(it1) != 0 and len(it2) != 0:
        x1 = it1[0]
        x2 = it2[0]

        if x1 <= x2:
            while x1 <= x2:
                i, x1 = _first_pair_within_radius(it1, x2, radius, dist_func=dist_func)
                if x1 is not None:
                    if inverted:
                        yield x2, x1
                    else:
                        yield x1, x2
                    it1 = it1[(i + 1):]
                    it2 = it2[1:]
                    if len(it1) == 0 or len(it2) == 0:
                        break
                    else:
                        x1 = it1[0]
                        x2 = it2[0]
                else:
                    break
        else:
            yield from _close_pairs_helper(it2, it1, radius=radius, inverted=not inverted, dist_func=dist_func)


def gen_of_close_pairs_from_iterables(it1, it2, radius=inf, dist_func=lambda x1, x2: abs(x1 - x2)):
    """Generate pairs of numbers taken from two iterators such that the values of the pairs are within a given distance

    Note: This function might be counter intuitive: You might want to use gen_nearest_neighbor_matching_pairs instead.
    Note: The elements will not be used twice
    Note: The pairs are NOT nearest neighbor pairs, but simply the first pairs within a radius of each other, found from
        a linear scan of the sorted elements.
    Args:
        it1: An iterable
        it2: Another iterable
        radius: maximum difference between elements of the pairs that will be generated

    >>> list(gen_of_close_pairs_from_iterables(range(20), [5, 10, 15], radius=0))
    [(5, 5), (10, 10), (15, 15)]
    >>> list(gen_of_close_pairs_from_iterables(range(20), [5, 10, 15], radius=1))
    [(4, 5), (9, 10), (14, 15)]
    >>> list(gen_of_close_pairs_from_iterables([4, 7, 8, 11], [5, 10, 15], radius=1))
    [(4, 5)]
    >>> list(gen_of_close_pairs_from_iterables([4, 7, 8, 11], [5, 10, 15], radius=2))
    [(4, 5), (8, 10)]
    >>> list(gen_of_close_pairs_from_iterables([5, 10, 15], [4, 7, 8, 11], radius=2))
    [(5, 4), (10, 8)]
    >>> list(gen_of_close_pairs_from_iterables([5, 10, 15], [4, 7, 8, 11], radius=3))
    [(5, 4), (10, 7)]
    """
    it1 = sorted(it1)
    it2 = sorted(it2)

    yield from _close_pairs_helper(sorted(it1), sorted(it2), radius, inverted=False, dist_func=dist_func)


def sublist_that_contains_segment(sorted_segment_mins, from_val=-inf, to_val=inf, key=None):
    """
    Returns the smallest segment-covering of an interval where the segments are partitions of the real line and are
    given (in the sorted_segment_mins) by a sorted list of smallest value of a segment.

     In some sense, it is the smallest sublist containing the [from_val, to_val] interval.
     We say "in some sense" because the function will range from the largest value strictly smaller than from_val to
     the smallest value strictly greater than to_val.
     For example, if sorted_segment_mins = [0, 5, 10, 15, 20, 25, 30], from_val=10, and to_val=20, the function will return
        [5, 10, 15, 20]
     as opposed to
        [10, 15, 20]
    This is useful when the sorted_segment_mins are actually taken from a larger sorted list and you want to make sure
    that the interval returned contains ALL vals that are greater OR EQUAL to from_val and smaller OR EQUAL to to_val.
    We call this the "superset-safe" property.
    :param sorted_segment_mins: A sorted iterator
    :param from_val: The smallest value to contain. If None will take from_val=-inf
    :param to_val: The largest value to contain. If None will take from_val=inf
    :param key: How to get the vals (to compare with) from the sorted_segment_mins iterator elements.
        * None: Will just take the element itself
        * callable: Will take key(element) as the val
        * else will take element[key] as the val (hoping that key is a hashable, and exists as a key to element)
    :return: A list of the elements of sorted_segment_mins.
    >>> sorted_segment_mins = [0, 5, 10, 15, 20, 25, 30]
    >>> sublist_that_contains_segment(sorted_segment_mins, 11, 21)
    [10, 15, 20]
    >>> # test the superset-safe property (see description above)
    >>> sublist_that_contains_segment(sorted_segment_mins, 10, 20)
    [5, 10, 15, 20]
    >>> # test specifying only from_val
    >>> sublist_that_contains_segment(sorted_segment_mins, from_val=11)
    [10, 15, 20, 25, 30]
    >>> # test specifying only to_val
    >>> sublist_that_contains_segment(sorted_segment_mins, to_val=21)
    [0, 5, 10, 15, 20]
    >>> # test a functional key
    >>> sorted_segment_mins = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 3, 'b': 4}, {'a': 4, 'b': 5}, {'a': 5, 'b': 6}]
    >>> sublist_that_contains_segment(sorted_segment_mins, 7, 12, key=lambda x: x['a'] * x['b'])
    [{'a': 2, 'b': 3}, {'a': 3, 'b': 4}]
    >>> # test a hashable key
    >>> sorted_segment_mins = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 3, 'b': 4}, {'a': 4, 'b': 5}, {'a': 5, 'b': 6}]
    >>> sublist_that_contains_segment(sorted_segment_mins, 2.1, 3, key='a')
    [{'a': 2, 'b': 3}, {'a': 3, 'b': 4}]
    """
    sublist = list()
    if from_val is None:
        from_val = -inf
    if to_val is None:
        to_val = inf
    if key is None:
        x_to_val = lambda x: x
    elif callable(key):
        x_to_val = key
    else:
        x_to_val = lambda x: x[key]

    previous_val = -inf
    for x in sorted_segment_mins:
        val = x_to_val(x)
        if val < previous_val:
            raise AssertionError("sorted_ts is not sorted")
        if val < from_val:
            sublist = [x]
        else:
            if val > to_val:
                break
            else:
                sublist.append(x)
        previous_val = val

    return sublist
