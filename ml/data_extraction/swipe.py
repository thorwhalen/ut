from ut.util.ulist import KeepMaxK


class HighestScoreSwipe(object):
    def __init__(self, score_of, chk_size, chk_step=1):
        self.score_of = score_of
        self.chk_size = chk_size
        self.chk_step = chk_step

    def __call__(self, it):
        pass


def highest_score_swipe(it, score_of=None, k=1, info_of=None, output=None):
    if score_of is None:
        score_of = lambda x: x

    km = KeepMaxK(k=k)

    if info_of is None:
        for x in it:
            km.push((score_of(x), x))
    else:
        if info_of == 'idx':
            for i, x in enumerate(it):
                km.push((score_of(x), i))
        else:
            assert callable(
                info_of
            ), "info_of needs to be a callable (if not None or 'idx')"
            for x in it:
                km.push((score_of(x), info_of(x)))

    if output is None:
        return km
    elif isinstance(output, str):
        if output == 'top_tuples':
            return sorted(km, reverse=True)
        elif output == 'items':
            return [x[1] for x in km]
        elif output == 'scores':
            return [x[0] for x in km]
        elif output == 'top_score_items':
            return [x[1] for x in sorted(km, key=lambda x: x[0])]
        else:
            raise ValueError('Unrecognized output: '.format(output))
    else:
        return km
