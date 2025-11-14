from ut.ml.sk.model_selection import SupervisedLeaveOneOut
from ut.util.log import printProgress
from numpy import array


def partial_leave_one_out_test(
    model, X, y, n_splits=None, min_n_samples_per_unik_y=None, verbose=1
):
    loo = SupervisedLeaveOneOut(
        n_splits=n_splits, min_n_samples_per_unik_y=min_n_samples_per_unik_y
    )
    n_splits = loo.get_n_splits()
    if verbose > 0:
        printProgress(f'Number of tests: {n_splits}')

    predicted = list()
    actual = list()
    for i, (train_idx, test_idx) in enumerate(loo.split(X, y), 1):
        XX = X[train_idx, :]
        yy = y[train_idx]
        if verbose > 0:
            printProgress(
                f'Test {i}/{n_splits}', refresh=True, refresh_suffix='   '
            )
        model.fit(XX, yy)
        test_x = X[test_idx, :]
        test_y = y[test_idx]
        actual.append(test_y[0])
        predicted.append(model.predict(test_x)[0])

    return array(predicted), array(actual)
