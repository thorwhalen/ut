__author__ = 'thor'

import pandas as pd
from numpy import argmax


def list_of_strings_from_df_rows(df, column_sep=';', str_cols=None):
    df = df.copy()

    # choose columns
    if str_cols is None:
        str_cols = []
        for c in df.columns:
            x = df[c].iloc[argmax(df[c].notnull())]
            if isinstance(x, str):
                str_cols.append(c)
                df[c].fillna('', inplace=True)
    else:
        str_cols = df.columns
        df.fillna('', inplace=True)
        for c in str_cols:
            df[c] = list(map(str, df[c]))

    return [column_sep.join(row[1]) for row in df[str_cols].iterrows()]