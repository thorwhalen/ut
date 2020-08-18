"""K-means clustering with weights"""
__author__ = 'sylvain'

import warnings

import numpy as np
import scipy.sparse as sp

# from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, squared_norm
# from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
# from sklearn.utils.fixes import astype
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import as_float_array
# from sklearn.utils import gen_batches
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
# from sklearn.utils.random import choice
from joblib import Parallel
from joblib import delayed
from six import string_types

import sys

if sys.version_info >= (3, 5):
    from sklearn.cluster import _k_means_fast as _k_means
else:
    from sklearn.cluster import _k_means


###############################################################################
# Initialization heuristic


def _k_init_with_weights(X, weights, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++, using weights
    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters: integer
        The number of seeds to choose

    weights: array of weights

    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state: numpy.RandomState
        The generator used to initialize the centers.
    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version modified to include weights.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features))

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init_with_weights'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly #####should use the weights here too
    center_id = np.random.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential

    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    closest_dist_sq_weighted = closest_dist_sq * weights  ##check if dimensions agree
    current_pot_weighted = closest_dist_sq_weighted.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot_weighted
        candidate_ids = np.searchsorted(closest_dist_sq_weighted.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)  # here we use the true distances
        distance_to_candidates_weighted = distance_to_candidates * weights  ##check syntax

        # Decide which candidate is the best
        best_candidate = None
        best_pot_weighted = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])  ###check here too
            # new_pot = new_dist_sq.sum()
            new_pot_weighted = (new_dist_sq * weights).sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot_weighted < best_pot_weighted):
                best_candidate = candidate_ids[trial]
                best_pot_weighted = new_pot_weighted
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot_weighted = best_pot_weighted
        closest_dist_sq = best_dist_sq
        ##include here also the updating of the weights
    return centers


###############################################################################
# K-means batch estimation by EM (expectation maximization)


def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))


def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def _labels_inertia_precompute_dense(X, x_squared_norms, centers, distances):
    """Compute labels and inertia using a full distance matrix.
    This will overwrite the 'distances' array in-place.
    Parameters
    ----------
    X : numpy array, shape (n_sample, n_features)
        Input data.
    x_squared_norms : numpy array, shape (n_samples,)
        Precomputed squared norms of X.
    centers : numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.
    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.
    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.
    inertia : float
        Sum of distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    k = centers.shape[0]
    all_distances = euclidean_distances(centers, X, x_squared_norms,
                                        squared=True)
    labels = np.empty(n_samples, dtype=np.int32)
    labels.fill(-1)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for center_id in range(k):
        dist = all_distances[center_id]
        labels[dist < mindist] = center_id
        mindist = np.minimum(dist, mindist)
    if n_samples == distances.shape[0]:
        # distances will be changed in-place
        distances[:] = mindist
    inertia = mindist.sum()
    return labels, inertia


def _labels_inertia(X, x_squared_norms, centers,
                    precompute_distances=True, distances=None):
    """E step of the K-means EM algorithm.
    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.
    Parameters
    ----------
    X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.
    x_squared_norms: array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.
    centers: float64 array, shape (k, n_features)
        The cluster centers.
    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).
    distances: float64 array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.
    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment
    inertia : float
        Sum of distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    # distances will be changed in-place
    if sp.issparse(X):
        inertia = _k_means._assign_labels_csr(
            X, x_squared_norms, centers, labels, distances=distances)
    else:
        if precompute_distances:
            return _labels_inertia_precompute_dense(X, x_squared_norms,
                                                    centers, distances)
        inertia = _k_means._assign_labels_array(
            X, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia


def _init_centroids_with_weights(X, weights, k, init, random_state=None, x_squared_norms=None,
                                 init_size=None):
    """Compute the initial centroids
    Parameters
    ----------
    X: array, shape (n_samples, n_features)

    weights: array, shape (n_samples)
    k: int
        number of centroids
    init: {'k-means++_with_weights', 'random' or ndarray or callable} optional
        Method for initialization
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    x_squared_norms:  array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None
    init_size : int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.
    Returns
    -------
    centers: array, shape(k, n_features)
    """
    random_state = check_random_state(random_state)  # will use np.random everywhere
    n_samples = X.shape[0]

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = np.random.random_integers(
            0, n_samples - 1, init_size)
        X = X[init_indices]
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
        raise ValueError(
            "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, string_types) and init == 'kmeans++_with_weights':
        centers = _k_init_with_weights(X, weights, k, random_state=random_state,
                                       x_squared_norms=x_squared_norms)
    elif isinstance(init, string_types) and init == 'random':
        seeds = np.random.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        centers = init
    elif callable(init):
        centers = init(X, k, random_state=random_state)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'kmeans++_with_weights' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if sp.issparse(centers):
        centers = centers.toarray()

    _validate_center_shape(X, k, centers)
    return centers


def _kmeans_single_with_weights(X, weights, n_clusters, x_squared_norms, max_iter=300,
                                init='kmeans++_with_weights', verbose=False, random_state=None,
                                tol=1e-4, precompute_distances=True):
    """A single run of k-means with weights, assumes preparation completed prior.
    Parameters
    ----------
    X: array-like of floats, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters: int
        The number of clusters to form as well as the number of
        centroids to generate.
    weights: array, shape (n_samples)
    max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    init: {'kmeans++_with_weights', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':
        'kmeans++_with_weights' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init_with_weights for more details.
        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.
        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.
        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.
    tol: float, optional
        The relative increment in the results before declaring convergence.
    verbose: boolean, optional
        Verbosity mode
    x_squared_norms: array
        Precomputed x_squared_norms.
    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).
    random_state: integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    Returns
    -------
    centroid: float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.
    label: integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    weighted inertia: float
        The final value of the inertia criterion (weighted sum of squared distances to
        the closest centroid for all observations in the training set).
    n_iter : int
        Number of iterations run.
    """
    # random_state = check_random_state(random_state)     ####remove this

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids_with_weights(X, weights, n_clusters, init, random_state=random_state,
                                           x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=np.float64)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, _ = \
            _labels_inertia(X, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

        # computation of the means is also called the M-step of EM
        # if sp.issparse(X):
        #   centers = _k_means._centers_sparse(X, labels, n_clusters,
        #                                      distances)
        # else:
        #    centers = _k_means._centers_dense(X, labels, n_clusters, distances)

        # computation of the weighted means is also called the M-step of EM ##Check this!!#
        weights_clusters = np.zeros(n_clusters)

        for i in range(len(X)):
            weights_clusters[labels[i]] += weights[i]

        n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)

        if sp.issparse(X):
            centers = _k_means._centers_sparse(np.multiply(X, weights.reshape(len(weights), 1)), labels, n_clusters,
                                               distances)
        else:
            centers = _k_means._centers_dense(np.multiply(X, weights.reshape(len(weights), 1)), labels, n_clusters,
                                              distances)

        weights_clusters = np.zeros(n_clusters)

        for i in range(len(X)):
            weights_clusters[labels[i]] += weights[i]
        weights_clusters[weights_clusters == 0] = 1
        n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
        centers /= weights_clusters[:, np.newaxis]
        centers *= n_samples_in_cluster[:, np.newaxis]

        # weighted inertia is computed here
        centers_labelled = centers[labels]
        row_norms_diff = row_norms(X - centers_labelled, squared=True)[np.newaxis, :]
        inertia = np.sum(weights * row_norms_diff)

        if verbose:
            print(("Iteration %2d, inertia %.3f" % (i, inertia)))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        shift = squared_norm(centers_old - centers)
        if shift <= tol:
            if verbose:
                print(("Converged at iteration %d" % i))

            break

    if shift > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, _ = \
            _labels_inertia(X, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
        # weighted best_inertia is computed here
        best_centers_labelled = best_centers[best_labels]
        best_row_norms_diff = row_norms(X - best_centers_labelled, squared=True)[np.newaxis, :]
        best_inertia = np.sum(weights * best_row_norms_diff)
    return best_labels, best_inertia, best_centers, i + 1


############without weights, just to compare, with some print statements within ######


def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++
    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters: integer
        The number of seeds to choose
    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state: numpy.RandomState
        The generator used to initialize the centers.
    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features))

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = np.random.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    # print 'initial center id= ', center_id
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    # print 'closest_dist_sq = '
    current_pot = closest_dist_sq.sum()
    # print 'current_pot='
    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        # print 'rand_vals', rand_vals
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)
        # print 'candidate ids=', candidate_ids

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        # print 'distance to candidates', distance_to_candidates

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


#########kmeans with weights###########


def k_means_with_weights(X, weights, n_clusters, init='kmeans++_with_weights', precompute_distances='auto',
                         n_init=10, max_iter=300, verbose=False,
                         tol=1e-4, random_state=None, copy_x=True, n_jobs=1,
                         return_n_iter=False):
    """K-means clustering algorithm with weights.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.
    weights : array of weights, shape (n_samples)
    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    init : {'kmeans++_with_weights', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++_with_weights':
        'k-means++_with_weights' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence.
        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    tol : float, optional
        The relative increment in the results before declaring convergence.
    verbose : boolean, optional
        Verbosity mode.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    copy_x : boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    return_n_iter : bool, optional
        Whether or not to return the number of iterations.
    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means_with_weights.
    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    inertia : float
        The final value of the inertia criterion (weighted sum of squared distances to
        the closest centroid for all observations in the training set).
    best_n_iter: int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    """
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    # random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    best_inertia = np.infty
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    # If the distances are precomputed every job will create a matrix of shape
    # (n_clusters, n_samples). To stop KMeans from eating up memory we only
    # activate this if the created matrix is guaranteed to be under 100MB. 12
    # million entries consume a little under 100MB if they are of type double.
    if precompute_distances == 'auto':
        n_samples = X.shape[0]
        precompute_distances = (n_clusters * n_samples) < 12e6
    elif isinstance(precompute_distances, bool):
        pass
    else:
        raise ValueError("precompute_distances should be 'auto' or True/False"
                         ", but a value of %r was passed" %
                         precompute_distances)

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X) or hasattr(init, '__array__'):
        X_mean = X.mean(axis=0)  # what about weighted means here?
    if not sp.issparse(X):
        # The copy was already done above
        X -= X_mean

    if hasattr(init, '__array__'):
        init = check_array(init, dtype=np.float64, copy=True)
        _validate_center_shape(X, n_clusters, init)

        init -= X_mean
        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None
    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = _kmeans_single_with_weights(
                X, weights, n_clusters, max_iter=max_iter, init=init, verbose=verbose,
                precompute_distances=precompute_distances, tol=tol,
                x_squared_norms=x_squared_norms, random_state=random_state)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:  # default n_jobs==1 should be used here
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_kmeans_single_with_weights)(X, weights, n_clusters, max_iter=max_iter,
                                                 init=init, verbose=verbose, tol=tol,
                                                 precompute_distances=precompute_distances,
                                                 x_squared_norms=x_squared_norms,
                                                 # Change seed to ensure variety
                                                 random_state=seed)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = list(zip(*results))
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class KMeansWeighted(object):
    """K-Means clustering with weights
    Read more in the :ref:`User Guide <k_means>`.
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    verbose : int, default 0
        Verbosity mode.
    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    Notes
    ------
    The k-means problem is solved using Lloyd's algorithm.

    """

    def __init__(self, n_clusters=8, init='kmeans++_with_weights', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = None
        self.copy_x = copy_x
        self.n_jobs = n_jobs

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                        warn_on_dtype=True)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X

    def fit(self, X, weights, y=None, debug=False):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        weights : array, shape (n_samples)
        """
        random_state = np.random  # random_state is always np.random
        if debug:
            print((type(X)))
            print(X)
            print((X.shape))
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means_with_weights(
                X, weights, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter,
                verbose=self.verbose, return_n_iter=True,
                precompute_distances=self.precompute_distances,
                tol=self.tol, random_state=self.random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs)
        return self

    def fit_predict(self, X, weights, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X, weights).labels_

    def fit_transform(self, X, weights, y=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        X = self._check_fit_data(X)
        return self.fit(X, weights)._transform(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]

    def score(self, X, y=None):
        """Opposite of the value of X on the K-means objective.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return -_labels_inertia(X, x_squared_norms, self.cluster_centers_)[1]


def main():
    pass


if __name__ == '__main__':
    main()
