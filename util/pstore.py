"""Utils to work with storage"""

import pandas as pd
import numpy as np
import ut.pcoll.order_conserving as colloc
import ut.pcoll.op as coll_op
import ut.util.ulist as util_ulist
import ut.daf.ch as daf_ch
import re
import ut.daf.get as daf_get
import ut.daf.manip as daf_manip
from pprint import PrettyPrinter
import pickle
import dill
import inspect
import gzip

__author__ = 'thorwhalen'

str_to_rep_by_nan = 'x9y7z1'  # hopefully this string never shows up


def gz_pickle_dump(object, filename, protocol=-1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    fp = gzip.GzipFile(filename, 'wb')
    pickle.dump(object, fp, protocol)
    fp.close()


def gz_pickle_load(filename):
    """Loads a compressed object from disk
    """
    fp = gzip.GzipFile(filename, 'rb')
    object = pickle.load(fp)
    fp.close()

    return object


def pickle_dump(obj, filepath=None, protocol=None):
    # make up a name for obj if filepath isn't given...
    if filepath is None:
        stack = inspect.stack()
        try:
            locals_ = stack[1][0].f_locals
        finally:
            del stack
        candidates = list()
        for k, v in list(locals_.items()):
            if v is obj:
                candidates.append(k)
        if candidates:
            # sort the candidates and take the last one (to avoid ipython cache (things like _40, _234, etc.)
            filepath = sorted(candidates)[-1] + '.p'
        else:  # still didn't find a name for the variable
            filepath = 'pickle.p'

    print(('Saving object to {}'.format(filepath)))
    try:
        return pickle.dump(obj, open(filepath, 'wb'), protocol=protocol or 0)
    except (ValueError, TypeError):
        return dill.dump(obj, open(filepath, 'wb'), protocol=protocol)


def pickle_load(filepath):
    try:
        return pickle.load(open(filepath, 'rb'))
    except (ValueError, AttributeError):
        return dill.load(open(filepath, 'rb'))


def store_names(store):
    store_info = dict()
    for k in list(store.keys()):
        print('  getting info for %s' % k)
        d = store.select(k, start=0, stop=1)
        store_info[k] = dict()
        store_info[k]['index_names'] = d.index.names
        store_info[k]['column_names'] = list(d.columns)
    return store_info


def ascertain_prefix_slash(key):
    if isinstance(key, str):
        if key[0] != '/':
            return '/' + key
        else:
            return key
    else:
        key = list(map(ascertain_prefix_slash, key))
        return key


def filepath(store):
    return store._path


def has_key(store, key):
    key = ascertain_prefix_slash(key)
    return key in list(store.keys())


def get_col_names(
    store,
    keys=None,
    singular_info='index_and_columns',
    print_results=False,
    style='dict',
):
    """

    :param store: a HDFStore
    :param keys: list of keys to get info from (if present)
    :return: a cols_info dict whose keys are the keys of the store and values are a dict with
    'index', 'columns', and 'index_and_columns' which contain the data col names
    """
    # process inputs
    if not keys:
        keys = list(store.keys())
    else:
        keys = util_ulist.ascertain_list(keys)
        keys = colloc.intersect(keys, list(store.keys()))
    # make a dict with col (and index) info
    cols_info = dict()
    for key in keys:
        cols_info[key] = dict()
        df = store[key]
        cols_info[key]['index'] = list(df.index.names)
        cols_info[key]['columns'] = list(df.columns)
        cols_info[key]['index_and_columns'] = (
            cols_info[key]['index'] + cols_info[key]['columns']
        )
    if singular_info:
        cols_info_copy = cols_info
        cols_info = dict()
        for key in keys:
            cols_info[key] = cols_info_copy[key][singular_info]
    if print_results:
        PrettyPrinter(indent=2).pprint(cols_info)
    if style == 'dataframe':
        d = pd.DataFrame()
        for k, v in cols_info.items():
            v = [x for x in v if x]
            d = pd.concat([d, pd.DataFrame(data=v, columns=[k])], axis=1)
        d = d.fillna(value='')
        cols_info = d.transpose()

    return cols_info


def get_info_df(store, keys=None, info=None, cols=None):
    # process inputs
    if not keys:
        keys = list(store.keys())
    else:
        keys = util_ulist.ascertain_list(keys)
        keys = colloc.intersect(keys, list(store.keys()))
    # get info_dict
    info_dict = get_info_dict(store)
    # make the df
    df = pd.DataFrame([dict(v, **{'key': k}) for k, v in info_dict.items()])
    df = df[df['key'].isin(keys)]
    if 'shape' in df.columns:
        del df['shape']
    if 'ncols' not in df.columns:
        df['ncols'] = np.nan
    if 'nrows' not in df.columns:
        df['nrows'] = np.nan
    # get ncols and nrows with missing
    idx = (
        df['ncols'].isnull().nonzero()[0]
    )  # ncols and nrows should both be missing when one is
    for i in idx:
        d = store[df['key'].iloc[i]]
        df['nrows'].iloc[i] = len(d)
        df['ncols'].iloc[i] = len(d.columns)
    # clean up and return
    df = df.set_index('key')
    df = df.sort_index()
    df = daf_manip.reorder_columns_as(
        df, ['nrows', 'ncols', 'isa', 'typ', 'indexers', 'dc']
    )
    df = df.replace(to_replace=np.nan, value='')
    if info:
        if isinstance(info, dict):
            # add as many columns as there are keys in dict, using the values of the dict as functions applied to
            # the whole stored dataframe to get the column value
            df = pd.concat(
                [df, pd.DataFrame(columns=list(info.keys()), index=df.index)], axis=1
            )
            for key in df.index.values:
                key_data = store[key]
                for k, v in info.items():
                    df[k].loc[key] = v(key_data)
        elif np.all([isinstance(x, str) for x in info]):
            df = daf_manip.filter_columns(df, info)
        else:
            raise ValueError('Unrecognized info format')
    # filter cols
    if cols:
        df = daf_manip.filter_columns(df, cols)
    return df


def get_info_dict(store, key=None):
    """
    :param store:
    :return: an info_dict with a bunch of information on the store
    """
    s = store.__repr__()
    t = re.split('\n', s)
    t = [re.split('\s+', x) for x in t[2:]]
    info_dict = dict()
    for ti in t:
        info_dict[ti[0]] = {'isa': ti[1], 'descr': ti[2][1:-1]}
    # parse the descr key
    for k, v in info_dict.items():
        descr = v['descr']
        # take out the dc-> part if present (because it can point to something with commas in it)
        m = re.findall('dc->\[[^\]]+\]', descr)
        if m:
            dc_list = m[0][5:-1].split(',')
            info_dict[k]['dc'] = dc_list
            descr = re.sub('dc->\[[^\]]+\]', '', descr)
        # take care of the remaining key->value
        m = re.findall('(\w+)->([^,]+)', descr)
        for ti in m:
            info_dict[k][ti[0]] = ti[1]
        del info_dict[k]['descr']
        # format the info
        if 'ncols' in list(info_dict[k].keys()):
            info_dict[k]['ncols'] = int(info_dict[k]['ncols'])
        if 'nrows' in list(info_dict[k].keys()):
            info_dict[k]['nrows'] = int(info_dict[k]['nrows'])
        if 'indexers' in list(info_dict[k].keys()):
            info_dict[k]['indexers'] = info_dict[k]['indexers'][1:-1].split(',')
    if key:
        if has_key(info_dict, key):
            return info_dict[ascertain_prefix_slash(key)]
        else:
            return dict()
    else:
        return info_dict


def copy_data(from_store, to_store, from_keys, overwrite=False):
    """
    Copies key contents from one store to another, overwriting or not (default), and respecting original store format.
    :param from_store: store (or path of store) to copy from
    :param to_store: store (or path of store) to copy to
    :param from_keys: list of keys to copy from from_store
    :param overwrite: if True, will remove existing key in to_store if they exist, if False, will not copy (silently)
    :return: None
    """
    # handle input formats
    if isinstance(from_store, str):
        from_store = MyStore(from_store)
        close_from_store = True
    else:
        close_from_store = False
    if isinstance(to_store, str):
        to_store = MyStore(to_store)
        close_to_store = True
    else:
        close_to_store = False
    from_keys = util_ulist.ascertain_list(from_keys)
    from_keys = list(map(ascertain_prefix_slash, from_keys))
    # if overwrite is False, keep only those keys that don't exist
    if not overwrite:
        from_keys = list(set(from_keys).difference(list(to_store.keys())))

    # get some info on the from_store
    store_info = get_info_dict(from_store)

    # do the copying
    for k in from_keys:
        store_df_respecting_given_format(
            to_store, k, from_store[k], key_info=store_info[k]
        )
    to_store.flush()

    # close stores (if they were specified by paths
    if close_from_store:
        from_store.close()
    if close_to_store:
        to_store.close()


def store_df_respecting_given_format(to_store, key, df, key_info=None):
    """
    Store data into a key respecting a given format
    :param to_store: Store to store to
    :param key: key to store to
    :param df: data to store
    :param key_info: information about the format to store as, which could be:
        * a dict containing format information
        * a dict (keyed by target key) of dicts of format information
        * a store from which to get the format information (by default, the target_store itself)
    :return:
    """
    # processing key_info (which could be a store, a dict of key_infos for a whole store, or the key_info itself
    if not key_info:  # if key_info not given, take it from to_store
        key_info = get_info_dict(to_store, key)
        if not key_info:
            raise ValueError(
                'either you have to give me key_info, or the key needs to be present in the target store'
            )
    elif (
        isinstance(key_info, dict)
        and has_key(key_info, key)
        and isinstance(key_info[key], dict)
    ):
        key_info = key_info[key]
    elif isinstance(key_info, pd.HDFStore):
        key_info = get_info_dict(key_info, key)
    # remove key in to_store if it exists already
    if key in list(to_store.keys()):
        to_store.remove(key)
    # store df in a specific format
    if key_info['isa'] == 'frame':
        to_store.put(key, df)
    else:
        if 'dc' in list(key_info.keys()):
            to_store.append(key, df, data_columns=key_info['dc'])
        else:
            to_store.append(key, df)


def copy_data_subset_from_specified_col_values(
    from_store, to_store, col, values, get_full_df_if_col_not_found=True
):
    store_info = get_info_dict(from_store)
    for key in list(from_store.keys()):
        # getting the data from from_store
        try:
            df = from_store[key]
        except:
            print("!!! couldn't get key %s in the source store" % key)
            # raise RuntimeWarning("couldn't get key %s in the source store" % key)
            continue
        # filtering in col==values
        df = daf_get.rows_with_col_values_in(
            df,
            col=col,
            values=values,
            return_full_df_if_col_not_found=get_full_df_if_col_not_found,
        )
        if len(df) > 0:
            # saving the data to to_store
            try:
                store_df_respecting_given_format(to_store, key, df, store_info[key])
            except Exception as e:
                print(
                    '!!! some problem occured when trying to put %s in the target store'
                    % key
                )
                print(e.message)
                # raise RuntimeWarning("some problem occured when trying to put %s in the target store" % key)
                continue
        else:
            print(
                "* there was no (filtered) data for key=%s, so I'm saving nothing for this key"
                % key
            )


class MyStore(pd.HDFStore):
    # def __init__(self, *args, **kwargs):
    #     print isinstance(self, MyStore)
    #     print type(self)
    #     self.as_super = super(MyStore, self)
    #     self.as_super.__init__(*args, **kwargs)
    #     #super(MyStore, self).__init__(*args, **kwargs)

    def filepath(self):
        return self._path

    def has_key(self, key):
        key = ascertain_prefix_slash(key)
        return key in list(self.keys())

    def mk_info_dict(self):
        return get_info_dict(self)

    def copy_data(self, to_store, from_keys, overwrite=False):
        copy_data(self, to_store, from_keys, overwrite=overwrite)

    def add_default_key(self, key):
        self.key = key

    def or_select_single_var(self, key=None, where=None):
        key = key or self.key
        # where_string = where[0]+'='
        return self.select(key=key, where=pd.Term(where[0], where[1]))

    def change_stored_data_to_appendable(self, key=None):
        key = key or self.key
        df = self[key]
        self.remove_and_append(key=key, value=df)
        self.flush()

    def replace(self, key=None, value=None):
        """
        This method stores data in an already existing key, using the same format
        (as in store_df_respecting_given_format() function)
        of the target key.
        :param key:
        :param value:
        :return:
        """
        key = key or self.key
        store_df_respecting_given_format(self, key, value)

    def remove_and_append(
        self, key=None, value=None, nan_rep=str_to_rep_by_nan, **kwargs
    ):
        key = key or self.key
        if value is None:
            raise ValueError("You're trying to save a None to a MyStore")
        else:
            try:
                # daf_ch.to_utf8(value, inplace=True)
                value = daf_ch.to_utf8(value)
            except:
                print('Failed to convert to utf8')
                UnicodeWarning('Failed to convert to utf8')
        # prepend slash to key if missing
        key = ascertain_prefix_slash(key)
        # replace nans with empty spaces
        value = daf_ch.replace_nans_with_spaces_in_object_columns(value)
        # if key exists, remove it's contents
        if key in list(self.keys()):
            self.remove(key=key)
        super(MyStore, self).append(key=key, value=value, nan_rep=nan_rep, **kwargs)

    def remove_and_put(self, key=None, value=None, **kwargs):
        key = key or self.key
        if value is None:
            raise ValueError("You're trying to save a None to a MyStore")
        else:
            try:
                # daf_ch.to_utf8(value, inplace=True)
                value = daf_ch.to_utf8(value)
            except:
                print('Failed to convert to utf8')
                UnicodeWarning('Failed to convert to utf8')
        # replace nans with empty spaces
        value = daf_ch.replace_nans_with_spaces_in_object_columns(value)
        # if key exists, remove it's contents
        if key in self:
            self.remove(key=key)
        super(MyStore, self).put(key=key, value=value, **kwargs)

    def remove_all_but(self, keys_not_to_remove):
        keys_not_to_remove = ascertain_prefix_slash(keys_not_to_remove)
        keys = list(self.keys())
        keys_to_remove = set(keys).difference(keys_not_to_remove)
        for key in keys_to_remove:
            self.remove(key)

    def put(self, key=None, value=None, **kwargs):
        key = key or self.key
        try:
            # daf_ch.to_utf8(value, inplace=True)
            value = daf_ch.to_utf8(value)
        except:
            print('Failed to convert to utf8')
            UnicodeWarning('Failed to convert to utf8')
        # replace nans with empty spaces
        value = daf_ch.replace_nans_with_spaces_in_object_columns(value)
        super(MyStore, self).put(key=key, value=value, **kwargs)

    def append(self, key=None, value=None, nan_rep=str_to_rep_by_nan, **kwargs):
        key = key or self.key
        try:
            # daf_ch.to_utf8(value, inplace=True)
            value = daf_ch.to_utf8(value)
        except:
            print('Failed to convert to utf8')
            UnicodeWarning('Failed to convert to utf8')
        # replace nans with empty spaces
        value = daf_ch.replace_nans_with_spaces_in_object_columns(value)
        super(MyStore, self).append(key=key, value=value, nan_rep=nan_rep, **kwargs)

    def head(self, key=None, nRows=5):
        key = key or self.key
        return self.select(key, start=0, stop=nRows)

    def select(self, *args, **kwargs):
        try:
            return super(MyStore, self).select(*args, **kwargs)
        except Exception as e:
            try:
                return super(MyStore, self).select(key=self.key, *args, **kwargs)
            except:
                raise e

    def keep(self, column, value_list, file_path):
        key_list = list(self.keys())
        sample_store = MyStore(file_path)
        for key in key_list:
            print('taking a sample from %s' % key)
            df = self[key]
            if column in df.columns:
                df = df[df[column].isin(value_list)]
            elif column in df.index.names:
                index_names = df.index.names
                df = df.reset_index()
                df = df[df[column].isin(value_list)]
                df = df.set_index(index_names)
            sample_store.put(key, df)
        sample_store.flush()
        sample_store.close()


class StoreSelector(pd.HDFStore):
    # Valid expressions
    # 'index>=date'
    # "columns=['A', 'D']"
    # "columns in ['A', 'D']"
    # 'columns=A'
    # 'columns==A'
    # "~(columns=['A','B'])"
    # 'index>df.index[3] & string="bar"'
    # '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    # "ts>=Timestamp('2012-02-01')"
    # "major_axis>=20130101"
    def __init__(self, store, key, selection_col, columns=None):
        try:
            super(StoreSelector, self).__init__(path=store, mode='r')
        except:
            self.__setattr__('_mode', 'r')
        print(store)
        super(StoreSelector, self).__init__(path=store, mode='r')
        self.key = key
        self.selection_col = selection_col
        self.columns = columns

    def get_table(self, selection, columns=None, key=None):
        key = key or self.key
        columns = columns or self.columns
        return self.select(
            key=self.key,
            where=pd.Term(self.selection_col, selection),
            columns=self.columns,
        )


class StoreAccessor(object):
    def __init__(self, **kwargs):
        self.store = dict()
        self.store_info = dict()
        self.add_from = dict()
        self.join_store = None
        self.join_key = None
        if 'store_path_dict' in list(kwargs.keys()):
            for k, v in list(kwargs['store_path_dict'].items()):
                print('processing store: %s' % k)
                self.store[k] = MyStore(v)
                self.store_info[k] = store_names(self.store[k])
        if 'add_from_dict' in list(kwargs.keys()):
            for k, v in list(kwargs['add_from_dict'].items()):
                self.add_from[k] = v
        if 'join_store' in list(kwargs.keys()):
            self.join_store = kwargs['join_store']
        if 'join_key' in list(kwargs.keys()):
            self.join_key = kwargs['join_key']
        if 'join_filter' in list(kwargs.keys()):
            self.join_filter = kwargs['join_filter']

    def join_col(
        self,
        df,
        add_cols,
        join_cols=None,
        join_key=None,
        join_store=None,
        join_filter=None,
        drop_joining_duplicates=True,
    ):
        """
        This function is meant to return the input df with add_cols added.
        These columns are fetched in join_store[join_key] and are aligned to df using join_cols.
        Note: At the time of this writing, only a restricted case is handled, namely:
            join_cols has only one element that must be in the index of the store
        """
        join_store = join_store or self.join_store
        join_key = join_key or self.join_key
        if isinstance(add_cols, str):
            if add_cols in list(self.add_from.keys()):
                if 'join_store' in list(self.add_from[add_cols].keys()):
                    join_store = join_store or self.add_from[add_cols]['join_store']
                if 'join_key' in list(self.add_from[add_cols].keys()):
                    join_key = join_key or self.add_from[add_cols]['join_key']
                if 'join_cols' in list(self.add_from[add_cols].keys()):
                    join_cols = join_cols or self.add_from[add_cols]['join_cols']
        join_cols = util_ulist.ascertain_list(join_cols)
        add_cols = util_ulist.ascertain_list(add_cols)
        # get the df values to join (and see if they're in cols or index)
        if coll_op.contains(list(df.columns), join_cols):
            df_join_cols_in_columns = True
            df_join_col_values = np.unique(df[join_cols])
        else:
            df_join_cols_in_columns = False
            df_join_col_values = np.unique(list(df.index))
        # get necessary information from store
        store_key_info = self.store_info[join_store]
        join_key = ascertain_prefix_slash(join_key)
        store_key_info = store_key_info[join_key]
        if len(join_cols) == 1 and join_cols[0] == 'index':
            print('uploading only specific indices for join_df')
            join_df = self.store[join_store].select(
                key=join_key,
                where=[pd.Term('index', df_join_col_values)],
                columns=add_cols,
            )
        elif join_cols in store_key_info['column_names']:
            print('uploading only specific columns for join_df')
            join_df = self.store[join_store].select(
                key=join_key,
                where=[pd.Term(join_cols[0], df_join_col_values)],
                columns=join_cols + add_cols,
            )
            join_df.set_index(join_cols[0])
        else:
            print('uploading the whole potential join_df')
            join_df = self.store[join_store].select(
                key=join_key, columns=join_cols + add_cols
            )
        # print join_cols
        # print add_cols
        # print join_df.head(10)
        # drop duplicates
        if drop_joining_duplicates == True:
            join_df = join_df.drop_duplicates()
        if coll_op.contains(list(join_df.columns), join_cols):
            join_df_cols_in_cols = True
        else:
            join_df_cols_in_cols = False
        # print df_join_cols_in_columns
        # print join_df_cols_in_cols
        # join
        if df_join_cols_in_columns:
            if join_df_cols_in_cols:
                return pd.merge(df, join_df, on=join_cols)
            else:
                return pd.merge(df, join_df, right_on=join_cols, left_index=True)
        else:
            if join_df_cols_in_cols:
                return pd.merge(df, join_df, right_index=True, left_on=join_cols)
            else:
                return pd.merge(df, join_df, right_index=True, left_index=True)
