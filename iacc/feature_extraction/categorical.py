

__author__ = 'thor'


def get_columns_with_max_and_min_nuniques(df,
                                          max_nuniques=None,
                                          min_nuniques=2,
                                          count_nans=True,
                                          min_non_nans=0):
    columns = df.columns
    n_rows, n_columns = df.shape

    if max_nuniques is None:
        max_nuniques = 0.5
    if max_nuniques < 1:  # if max_nuniques is less then 1, consider it as a proportion of n_rows
        max_nuniques *= n_rows
    if min_non_nans < 1:
        min_non_nans *= n_rows
    assert min_nuniques <= max_nuniques, "You should have min_nuniques <= max_nuniques"

    column_choices = []
    for c in columns:
        t = df[c]
        not_null_lidx = t.notnull()
        if not count_nans:
            t = t[not_null_lidx]
        n = t.nunique()
        if n >= min_nuniques:
            if n <= max_nuniques:
                if min_non_nans > 0:
                    if sum(not_null_lidx) > min_non_nans:
                        column_choices.append(c)
                else:
                    column_choices.append(c)

    return column_choices



