from numpy import where, array
from numpy.random import randint, rand
import json
from ut.util.uiter import (
    indexed_sliding_window_chunk_iter,
    _inefficient_indexed_sliding_window_chunk_iter,
)

indexed_chunking_f_list = (
    _inefficient_indexed_sliding_window_chunk_iter,
    indexed_sliding_window_chunk_iter,
)


def rand_factor(ratio_radius=3):
    """
    Get a random number between (1/ratio_radius) and ratio_radius.
    This will be used to multiply with number (the "center") to get a number bigger or smaller
    than it.
    """
    ratio_diameter = ratio_radius - (1.0 / ratio_radius)
    return (1.0 / ratio_radius) + rand() * ratio_diameter


def random_kwargs_for_list(x, key=None, return_tail=False):
    if key is None:
        key = lambda x: x
    assert sorted(x) == x, 'x must be sorted'
    n = len(x)
    chk_size = randint(1, 100)
    chk_step = max(1, int(chk_size * rand_factor()))
    start_at = key(x[0]) * rand() * rand_factor()
    stop_at = key(x[-1]) * rand() * rand_factor()
    if start_at > stop_at:
        start_at, stop_at = stop_at, start_at  # just reverse them
    return dict(
        chk_size=chk_size,
        chk_step=chk_step,
        start_at=start_at,
        stop_at=stop_at,
        key=key,
        return_tail=return_tail,
    )


def indexed_chunking_random_test(
    f_list=indexed_chunking_f_list, x=None, return_debug_info=False, verbose=0
):
    """made it so you can just run a function (several times) to test, but if you want to see print outs use verbose=1,
    and if you want to get a bunch of variables that will then allow you to diagnose things,
    specify return_debug_info=True"""
    if x is None:
        x = randint(10, 1000)
    if isinstance(x, int):
        n_pts = x
        x = sorted(randint(1, 100000, n_pts))
    assert sorted(x) == x, 'x is not sorted!'

    kwargs = random_kwargs_for_list(x)
    if verbose:
        print(('x: {} elements. min: {}, max: {}'.format(len(x), x[0], x[-1])))
    t = {k: v for k, v in kwargs.items() if k != 'key'}
    if verbose:
        print(('kwargs: {}\n'.format(json.dumps(t, indent=2))))

    b = list(f_list[0](iter(x), **kwargs))
    bb = None
    all_good = True
    idx_where_different = array([])
    for i, f in enumerate(f_list[1:], 1):
        bb = list(f(iter(x), **kwargs))
        all_good = True
        if len(b) != len(bb):
            all_good &= False
            if verbose:
                print(
                    (
                        '{}: Not the same length! Base had {} elements, comp has {}'.format(
                            i, len(b), len(bb)
                        )
                    )
                )
        idx_where_different = where([x[0] != x[1] for x in zip(b, bb)])[0]
        if len(idx_where_different) > 0:
            all_good &= False
            if verbose:
                print(('{} values where different'.format(len(idx_where_different))))
        if not all_good:
            if verbose:
                print('STOPPING HERE: Check the variables for diagnosis')
            break
        print('')
    if all_good:
        if verbose:
            print('All good!')
    if return_debug_info:
        return all_good, idx_where_different, x, b, bb, kwargs
    else:
        return all_good


if __name__ == '__main__':
    for i in range(100):
        assert indexed_chunking_random_test(), 'indexed_chunking_random_test failed'
