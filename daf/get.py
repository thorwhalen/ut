__author__ = "thorwhalen"

import pandas as pd
import numpy as np
import string
import random
from ut.pfile.name import fileparts, data_file, delim_file
from ut.util import log
import pickle
import zipfile
import gzip
import re


def lidx_of_rows_whose_col_values_match_pattern(df, col, pattern):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    return df[col].isin(list(filter(pattern.match, df[col])))


def rows_whose_col_values_match_pattern(df, col, pattern):
    return df[lidx_of_rows_whose_col_values_match_pattern(df, col, pattern)]


def from_zip(zipfilename, data_reader=pickle.load, **kwargs):
    """
    loads a dataframe from a zip file
    """
    z = zipfile.ZipFile(zipfilename)
    if "filename" in list(kwargs.keys()):
        filename = kwargs["filename"]
    else:
        filename = z.filelist
        if len(filename) > 1:
            raise ValueError(
                "%s had several archived files: You need to specify which you want!"
                % zipfilename
            )
        else:
            filename = filename[0]
    return data_reader(z.open(filename), **kwargs)


def from_gzip(zipfilename, data_reader=pickle.load, **kwargs):
    """
    loads a dataframe from a gzip file
    """
    return data_reader(gzip.open(zipfilename), **kwargs)


def free_col_name(df, candidate_cols, raise_error=True):
    """
    Will look for the first string in candidate_cols that is not a column name of df.
    If no free column is found, will raise error (default) or return None.
    """
    for col in candidate_cols:
        if col not in df.columns:
            return col
    if raise_error:
        ValueError("All candidate_cols were already taken")
    else:
        return None


def get_unique_values(df, col):
    if col in df.columns:
        return list(np.unique(df[col]))
    elif col in df.index.names:
        return list(np.unique(df.index.get_level_values(col)))
    else:
        return list()


def get_values(df, col):
    if col in df.columns:
        return list(df[col])
    elif col in df.index.names:
        return list(df.index.get_level_values(col))
    else:
        return list()


def ilocs_of_rows_with_any_nans(df):
    return pd.isnull(df).any(1).nonzero()[0]


def ilocs_of_rows_with_all_nans(df):
    return pd.isnull(df).all(1).nonzero()[0]


def rows_with_col_values_in(df, col, values, return_full_df_if_col_not_found=True):
    col_names = df.columns
    if col in col_names:
        return df[df[col].isin(values)]
    else:
        index_names = df.index.names
        if col in index_names:
            df = df.reset_index()
            df = df[df[col].isin(values)]
            return df.set_index(keys=index_names)
        else:
            if return_full_df_if_col_not_found:
                return df
            else:
                return df.iloc[0:0].copy()


def column_names_whose_values_are_instances_of(df, instance_name):
    return [c for c in df.columns if isinstance(df[c].iloc[0], instance_name)]


def column_types(df):
    """
    returns a dict whose keys are the columns of df and whose values are the types of the elements of these columns.
    Note: only the first element of the column will be checked to determine the type
    """
    df = pd.DataFrame()
    df.to_excel()
    return {
        k: v
        for (k, v) in zip(
            df.columns, list(map(type, [df[c].iloc[0] for c in df.columns]))
        )
    }


def unique(d, cols=None):
    if cols is None:
        cols = d.columns.tolist()
    d = d.reindex(index=list(range(len(d))))
    grouped = d.groupby(cols)
    index = [gp_keys[0] for gp_keys in list(grouped.groups.values())]
    return d.reindex(index)


def get_data(dat, data_folder=""):
    # input: dat (a csv pfile, a data pfile, or the data itself
    # output: load data
    if isinstance(dat, str):  # if input dat is a string
        root, name, ext = fileparts(dat)
        if root:  # if root is not empty
            data_folder = root
        dataFile = data_file(dat, data_folder)
        if dataFile:
            df = pd.load(dataFile)
        else:
            delimFile = delim_file(dat)
            if delimFile:
                log.printProgress("csv->DataFrame")
                df = pd.read_csv(delimFile)
            else:
                raise NameError("FileNotFound")
        return df
    else:  # assume isinstance(dat,pd.DataFrame) or isinstance(dat,pd.Series)
        return dat


def mk_series(df, indexColName, dataColName):
    df = get_data(df)
    sr = df[dataColName]
    sr.index = df[indexColName].tolist()
    return sr


def duplicates(df, cols):
    df = df.reindex(index=list(range(len(df))))
    grouped = df.groupby(cols)
    unique_index = [gp_keys[0] for gp_keys in list(grouped.groups.values())]
    non_unique_index = list(set(df.index) - set(unique_index))
    duplicates_df = unique(df.irow(non_unique_index), cols)
    duplicates_df = duplicates_df[cols]
    return df.merge(pd.DataFrame(duplicates_df))


def rand(nrows=9, ncols=None, values_spec=None, columnTypes=None, columns=None):
    """
    returns a random dataframe
    Example: df = rand(nrows=10,ncols=3,columnTypes=['int','float','char'])
    """
    # argument processing
    if ncols is None:
        if isinstance(columnTypes, list):
            ncols = len(columnTypes)
        elif isinstance(columns, list):
            ncols = len(columns)
        elif isinstance(values_spec, list):
            ncols = len(values_spec)
        else:
            ncols = 3
    if values_spec is None:
        values_spec = max(2, int(np.ceil(nrows / 2)))
    if not isinstance(values_spec, list):
        values_spec = [values_spec for x in range(ncols)]
    if columnTypes is None:
        columnTypes = "int"  # possible types: 'int','float', or 'char'
    if not isinstance(columnTypes, list):
        columnTypes = [columnTypes for x in range(ncols)]
    if columns is None:
        columns = list(string.ascii_uppercase[:ncols])

    values_list = []
    for i in range(ncols):
        if hasattr(values_spec[i], "__getitem__"):
            n_values = len(values_spec[i])
            new_values_idx = np.random.randint(0, n_values, nrows)
            new_values = list(map(values_spec[i].__getitem__, new_values_idx))
        else:
            if columnTypes[i] == "int" or columnTypes[i] == int:
                new_values = np.random.randint(1, values_spec[i] + 1, nrows)
            elif columnTypes[i] == "float" or columnTypes[i] == float:
                new_values = np.random.rand(nrows) * (values_spec[i] - 1)
            elif columnTypes[i] == "char":
                assert (
                    values_spec[i] <= 26
                ), "can only get a max of 26 distinct chars at this point"
                letter_idx = np.random.randint(0, values_spec[i] - 1, nrows)
                new_values = [string.ascii_lowercase[:10][x] for x in letter_idx]
            elif columnTypes[i] in [str, str, str, "str", "string"]:
                word_bag = np.array(
                    [
                        "".join(random.choice(string.ascii_lowercase) for i in range(3))
                        for i in range(values_spec[i])
                    ]
                )
                new_values = word_bag[np.random.randint(0, values_spec[i], nrows)]
                # new_values = [''.join(random.choice(string.lowercase) for i in range(3)) for i in range(nrows)]
            else:
                raise ValueError(
                    "columnTypes must be 'float','int', 'char', 'string', 'str', "
                    "float, int, str, unicode, or basestring"
                )
        values_list.append(list(new_values))
    df = pd.DataFrame(values_list).transpose()
    df.columns = columns
    return df
    # return pd.DataFrame(new_values,columns=columns)
    # if isinstance(values_spec,int):
    #     return pd.DataFrame(np.random.randint(0,values_spec-1,[nrows,ncols]),columns=columns)
    # else:
    #     return pd.DataFrame(np.random.rand([nrows,ncols]),columns=columns)
    # # ''.join([string.lowercase[:10][x] for x in np.array([1,1,2,3,2,4,4,3])])


rand_df = rand
rand_df.__name__ = "rand_df"


def chunks(df, chk_size):
    """
    A generator that yields a dataframe in chunks
    """
    for i in range(0, len(df), chk_size):
        if i + chk_size < len(df):
            yield df.iloc[list(range(i, i + chk_size))]
        else:
            yield df.iloc[list(range(i, len(df)))]
