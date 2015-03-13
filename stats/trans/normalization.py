__author__ = 'thor'


from pandas import DataFrame
from operator import sub
from numpy import mean


def group_normalization(df, var_col, group=None, agg=mean, dif=sub,
                        keep_anchor=False, anchor_name='anchor'):
    """
    returns a dataframe where the var_col column values have been normalized relative to an aggregation of the values
    in a same group.
        * var_col: the name of the column to be normalized
        * group: will be fed to df.groupby(.) (usually list of columns names that define the groups)
        * agg: Will be fed to the df.groupby().agg() function
            - function to aggregate the (var_col) values of a same group to
            get an anchor for that group
            - list of {var_col: [(agg_name, agg_function)]} to create anchor(s) from var_col(s)
        * dif: A two argument function (x,y) that will normalize x according to anchor y.
        This function's inputs x and y should be ITERABLES. You can use numpy.normalize to make a function
         that works with iterables from a function that works with singular elements
        * keep_anchor: boolean specifying whether to keep the anchor column (default False)
        * anchor_name: a name for the anchor value, when singular

    >> t = ms.daf.get.rand(nrows=4, values_spec=[2,1,9], columns=['A','B','C'])
    >> print t
    >> print group_normalization(t, 'C', agg=sum, dif=lambda x, y: x - y, keep_anchor=True)
       A  B  C
    0  2  1  1
    1  2  1  7
    2  1  1  9
    3  1  1  1
       A  B  C  anchor
    0  2  1 -7       8
    1  2  1 -1       8
    2  1  1 -1      10
    3  1  1 -9      10

    """

    # prepare: If not explicitly given group_cols to be all non-var_col columns
    if group is None:
        group = list(set(df.columns).difference([var_col]))

    # group
    dg = df.groupby(group)

    # aggregate
    if hasattr(agg, '__call__'):
        dg = dg.agg({var_col: [(anchor_name, agg)]})
        dg.columns = dg.columns.droplevel(level=0)
        anchor_cols = anchor_name
    else:
        dg = dg.agg(agg)
        dg.columns = dg.columns.droplevel(level=0)
        anchor_cols = list(set(dg.columns).difference(df.columns))  # list of new column names


    # normalize (combining original values and group aggregates)
    dg.reset_index(drop=False, inplace=True)
    dg = df.merge(dg, on=group)
    # try:
    dg[var_col] = dif(dg[var_col], dg[anchor_cols])
    # except KeyError:
    #     dg[var_col] = dif(dg[var_col], dg[anchor_cols[0]])

    # clean up
    if not keep_anchor:
        dg.drop(labels=anchor_cols, axis=1, inplace=True)
    return dg

