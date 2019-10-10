__author__ = 'thorwhalen'

import json


def keyval_df(df, key_col=None, val_col=None, warn=False):
    # setting defaults
    if key_col is None:
        if 'key' in df.columns:
            key_col = 'key'
        else:
            key_col = df.columns[0]
            if warn:
                print("!!! using %s as the key_col" % key_col)
    if val_col is None:
        if 'val' in df.columns:
            val_col = 'val'
        else:
            val_col = df.columns[1]
            if warn:
                print("!!! using %s as the val_col" % val_col)
    return {k:v for (k,v) in zip(df[key_col],df[val_col])}


def json_str_file(filepath):
    return json.loads(json.load(open(filepath, 'r')))

