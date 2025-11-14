from numpy import reshape, ones, allclose, tile, random, vstack, array, isnan, all, any
import re
from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.exceptions import NotFittedError

__author__ = 'thor'


fitted_attribute_pattern = re.compile('.*_$')


def is_fitted_for_nfeats(model, nfeats):
    try:
        if hasattr(model, 'predict'):
            model.predict(random.rand(3, nfeats))
        elif hasattr(model, 'transform'):
            model.transform(random.rand(3, nfeats))
        return True
    except NotFittedError as e:
        return False


def is_fitted(model, attributes=None, all_or_any=all):
    """
    Returns True if the model is fitted, and False otherwise.

    How does the function know the model is fitted?

    If no attributes are given (the default), it will check for the
    existence of any attribute ending with an underscore. This is a bit of a risky non-explicit check, but can be
    useful. Use a more explicit check if needed.

    If attributes is not None, the model, attributes, and all_or_any arguments will be passed on to
    sklearn.utils.validation.check_is_fitted and return True if no NotFittedError was raised, and False if not.
    """
    if attributes is None:
        # return True if and only if model has at least one attribute ending with an underscore
        # print(model.__dict__.keys())
        for attr in model.__dict__:
            if fitted_attribute_pattern.match(attr):
                return True
        return False
    else:
        print('boo')
        try:
            check_is_fitted(model, attributes, all_or_any=all_or_any)
            return True
        except NotFittedError:
            return False


def weighted_data(X):
    """
    Takes an object X and returns a tuple X, w where X is a (n_samples, n_features) array and w is a (nsamples,) array
    corresponding to weights of the rows of X.

    X, w = weighted_data(X) is a convinience function to get weighted data.

    If the input X is just an array, it will consider it to be the data X, and will return w as all ones
    (aligned to the number of rows)

    If the input X is a tuple (X, w), it will check that w is aligned with the rows of X, and return the same X, w if so.

    """
    if isinstance(X, tuple):
        if len(X) == 1:
            X = X[0]
            if len(X.shape) == 1:
                X = reshape(X, (len(X), 1))
            elif len(X.shape) > 2:
                raise ValueError('data must be a matrix with no more than 2 dimensions')
            w = ones(X.shape[0])
        elif len(X) > 2:
            raise ValueError('X must be a 2-tuple of data (matrix) and weights')
        else:
            X, w = X
    else:
        w = ones(X.shape[0])

    return X, w


def abs_ratio_close(a, b, rtol=1e-05, atol=1e-08):
    t = abs(a / b)
    t = t[~isnan(t)]
    return allclose(t, 1.0, rtol=rtol, atol=atol)


def compare_model_attributes(
    model_1,
    model_2,
    exclude_attr=None,
    only_attr=None,
    msg_prefix='',
    close_enough_fun=allclose,
):
    """
    compare_model_attributes(model_1, model_2) is a convenience function to test if the model attributes
    (the attributes that are created and populated by the fit method of an sklearn model) of both models are the same
    (or, really, close enough (using numpy's allclose function)).

    It doesn't return anything, it just prints messages saying what attributes were not close enough to equal,
    or an "all fitted attributes were close" message if all was good.
    """
    if only_attr is not None:
        if isinstance(only_attr, str):
            only_attr = [only_attr]
        model_attributes_to_check = only_attr
    else:
        model_attributes_to_check = {
            attr
            for attr in list(model_1.__dict__.keys())
            if attr.endswith('_') and attr in model_2.__dict__
        }
    if exclude_attr is not None:
        if isinstance(exclude_attr, str):
            exclude_attr = [exclude_attr]
        model_attributes_to_check = {
            attr for attr in model_attributes_to_check if attr not in exclude_attr
        }

    not_close_attribs = list()
    for attr in model_attributes_to_check:
        try:
            assert close_enough_fun(getattr(model_1, attr), getattr(model_2, attr)), (
                msg_prefix
                + '{} of {} and {} not close'.format(
                    attr, model_1.__class__, model_2.__class__
                )
            )
        except AssertionError as e:
            not_close_attribs.append(attr)
    if len(not_close_attribs) > 0:
        print(msg_prefix + "Fitted attributes whose values weren't close enough:")
        print('  ' + '\n  '.join(not_close_attribs))
    else:
        print(msg_prefix + 'all fitted attributes were close')


def repeat_rows(X, row_repetition=None):
    """
    XX = repeated_rows(X, w) takes a data matrix X and an array w of len(X) elements (same number of rows as X).
    w should really be an array of ints, if they're not, they'll be rounded to be so.

    The function returns a data matrix XX that was constucted by repeating the ith row X[i, :] of X w[i] times,
    and concatinating the results.

    This is a convenient function to test weighted data models, since if model_2 is a "weighted model"
    version of model_1, then you should get the same thing with model_1.fit(repeated_rows(X))
    as you do with model_2.fit((X, w)).
    """
    if row_repetition is None:
        row_repetition = random.rand(1, 5, len(X))
    row_repetition = array(row_repetition).astype(int)
    return vstack(
        [
            tile(row_and_weight[0], (row_and_weight[1], 1))
            for row_and_weight in zip(X, row_repetition)
        ]
    )
