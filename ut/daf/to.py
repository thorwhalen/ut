__author__ = 'thor'

import pandas as pd
import os
import subprocess
from bs4 import BeautifulSoup
import copy
import itertools
import numpy as np
from hashlib import md5

import ut as ms
from ut.daf.manip import recursive_update
from ut.util.ulist import sort_as
from ut.daf.resources.disp_templates import inline_html_table_template

from collections import OrderedDict


def sr_to_ordered_dict(sr):
    return OrderedDict(iter(sr.items()))


def map_strings_to_their_mp5s(df, cols_to_map):
    if isinstance(cols_to_map, str):
        cols_to_map = [cols_to_map]

    def md5_of(s):
        m = md5()
        m.update(s)
        return m.hexdigest()

    df = df.copy()
    for c in cols_to_map:
        df[c] = df[c].apply(md5_of)

    return df


def map_vals_to_ints_inplace(df, cols_to_map):
    """
    map_vals_to_ints_inplace(df, cols_to_map) will map columns of df inplace
    If cols_to_map is a string, a single column will be mapped to consecutive integers from 0 onwards.
    Elif cols_to_map is a dict, the function will attempt to map the columns dict.keys() using the (list-like)
        maps specified in dict.values()
    Else assumed that cols_to_map is list-like specifying columns to be mapped

    Note: If no mapping is specified (i.e. cols_to_map is not a dict), the function will return the mapping dict,
    which is of the form mapping_dict[col][val] := mapped val for col.
    This can be used to reverse the operation, i.e.:
        mapping_dict = map_vals_to_ints_inplace(df, cols_to_map)
        map_vals_to_ints_inplace(df, mapping_dict)
        will leave df unchanged
    This can also be used to apply the same mapping to multiple columns. To apply the same mapping to A and B:
        mapping_dict = map_vals_to_ints_inplace(df, ['A'])
        map_vals_to_ints_inplace(df,
            cols_to_map={'B': dict(zip(mapping_dict['A'], range(len(mapping_dict['A']))))})

    """
    mapping_dict = dict()

    if isinstance(cols_to_map, str):  # mapping a single column
        mapping_dict, df[cols_to_map] = np.unique(df[cols_to_map], return_inverse=True)
        mapping_dict = {cols_to_map: mapping_dict}
    elif isinstance(cols_to_map, dict):  # mapping with a user specified map
        assert set(cols_to_map.keys()).issubset(
            df.columns
        ), 'cols_to_map keys must be a subset of df columns'
        for c in list(cols_to_map.keys()):
            this_map = cols_to_map[c]
            if isinstance(this_map, dict):
                assert all(
                    np.unique(list(this_map.values()))
                    == np.array(list(range(len(this_map))))
                ), 'you must map to consecutive integers starting at 0'
                df[c] = df[c].apply(lambda x: this_map[x])
                mapping_dict[c] = sort_as(
                    list(this_map.keys()), list(this_map.values())
                )
            else:
                df[c] = np.array(this_map)[list(df[c])]
                mapping_dict[c] = this_map
    else:  # mapping multiple columns
        for c in cols_to_map:
            mapping_dict.update(map_vals_to_ints_inplace(df, c))
    # return the mapping dict
    return mapping_dict


def to_html(df, template='hor-minimalist-b', template_overwrites=None, **kwargs):
    """
    enhancing pandas' to_html function to format html nicer
    """
    # getting the template
    if isinstance(template, str):
        template = copy.deepcopy(inline_html_table_template[template])
    # making some requested template changes
    recursive_update(template, template_overwrites)
    # serializing the styles
    for k in list(template.keys()):
        if 'style' in list(template[k].keys()):
            if not isinstance(template[k]['style'], str):
                template[k]['style'] = '; '.join(
                    [kk + ': ' + v for kk, v in list(template[k]['style'].items())]
                )
    # start with the pandas generated html (soup)
    b = BeautifulSoup(df.to_html(**kwargs))
    # update table formatting
    recursive_update(b.find('table').attrs, template['table'])
    # update thead.th formatting
    [
        recursive_update(x.attrs, template['thead'])
        for x in b.find('thead').find_all('th')
    ]
    # update tboday.tr.td formatting
    [
        recursive_update(x.attrs, template['tbody'])
        for x in list(
            itertools.chain.from_iterable(
                [
                    x.find_all('td')
                    for x in [xx for xx in b.find('tbody').find_all('tr')]
                ]
            )
        )
    ]
    return b.renderContents()


def insert_in_mongdb(df, collection, delete_previous_contents=False, **kwargs):
    """
    insert the rows of the dataframe df (as dicts) in the given collection.
    If you want to do it given a mongo_db and a collection_name:
        insert_in_mongdb(df, getattr(mongo_db, collection_name), **kwargs):
    If you want to do it given (a client, and...) a db name and collection name:
        insert_in_mongdb(df, getattr(getattr(client, db_name), collection_name), **kwargs):
    """
    if delete_previous_contents:
        collection_name = collection.name
        mother_db = collection.database
        mother_db.drop_collection(collection_name)
        mother_db.create_collection(collection_name)
    kwargs = dict(kwargs, **{'safe': True})  # default is safe=True
    collection.insert(dict_list_of_rows(df), **kwargs)


### Don't know where I was inspired by this version below, but anyway, I think my newer version is simpler and faster
# def insert_in_mongdb(df, mongo_db, collection_name, ubiquitous_dict=None, **kwargs):
#     kwargs = dict(kwargs, **{'safe': True})
#     mdb_collection = getattr(mongo_db, collection_name)
#     if ubiquitous_dict is None:
#         for i in range(len(df)):
#             mdb_collection.insert(df.iloc[i].to_dict(), **kwargs)
#     elif isinstance(ubiquitous_dict, dict):
#         for i in range(len(df)):
#             mdb_collection.insert(dict(ubiquitous_dict, **df.iloc[i].to_dict()), **kwargs)
#     else:
#         raise TypeError("I don't know this type of ubiquitous_dict!!")


def dict_list_of_rows(df, dropna=False):
    if dropna:
        return [
            {k: v for k, v in x.items() if not isinstance(v, float) or not np.isnan(v)}
            for x in df.transpose().to_dict().values()
        ]
    else:
        return [x for x in df.transpose().to_dict().values()]
    # is 1.62 times faster than [dict(row) for i, row in df.iterrows()] in case you were wondering


def dict_list_of_cols(df):
    DeprecationWarning('Come on! Use df.to_dict() directly would you?!')
    return df.to_dict()


def zip_pickle(df, filepath):
    df.to_pickle(filepath)
    zippath = ms.pfile.to.zip_file(filepath)
    os.remove(filepath)
    return zippath


def gzip_pickle(df, filepath):
    df.to_pickle(filepath)
    subprocess.check_call(['gzip', filepath])
    os.remove(filepath)
    return filepath + '.gzip'


def excel_multiple_sheets(
    df_enum, xls_filepath='excel_multiple_sheets.xlsx', sheet_names=None, **kwargs
):
    """
    Saves multiple dataframes (or one if you want) to multiple sheets of a same excel file
    Works with an enumeration (so could come from a generator) of dataframes.
    """

    if isinstance(df_enum, pd.DataFrame):
        df_enum = [df_enum]
    # choose different defaults than pd.to_excel
    kwargs = dict({'index': False}, **kwargs)
    # write the dataframes
    excel_writer = pd.ExcelWriter(xls_filepath)
    for n, df in enumerate(df_enum):
        if sheet_names is None:
            this_sheet_name = 'Sheet%s' % n
        else:
            this_sheet_name = sheet_names[n]
        df.to_excel(excel_writer=excel_writer, sheet_name=this_sheet_name, **kwargs)
    excel_writer.save()
    excel_writer.close()
