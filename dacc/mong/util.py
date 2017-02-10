__author__ = 'thor'

import pymongo as mg
from pymongo import MongoClient
from pymongo.errors import CursorNotFound

import os
import re
import pandas as pd
from numpy import inf, random, int64, int32, ndarray, float64, float32
import subprocess
from datetime import datetime

from ut.daf.to import dict_list_of_rows
from ut.daf.manip import rm_cols_if_present
from ut.daf.manip import rollout_cols
from ut.serialize.s3 import S3
from ut.pfile.to import ungzip
from ut.util.log import printProgress
import numpy as np
from ut.pdict.manip import recursively_update_with
from ut.util.pobj import inject_method
from pymongo.collection import Collection

s3_backup_bucket_name = 'mongo-db-bak'

# import boto
# import boto.s3
#
#

# # Fill these in or put them as environment variables - you get them when you sign up for S3
# try:
#     AWS_ACCESS_KEY_ID = os.environ['VEN_AWS_ACCESS_KEY_ID']
#     AWS_SECRET_ACCESS_KEY = os.environ['VEN_AWS_SECRET_ACCESS_KEY']
# except KeyError:
#     pass



def missing_indices(mgc, required_indices_keys):
    if isinstance(required_indices_keys, dict):  # assume it's the direct output of a collection.index_information()
        # get the keys of required_indices_keys
        required_indices_keys = keys_of_index_information(required_indices_keys)

    mgc_index_info = set(map(tuple, keys_of_index_information(mgc.index_information())))

    missing_keys = list()
    for k in required_indices_keys:
        if tuple(k) not in mgc_index_info:
            missing_keys.append(k)

    return missing_keys


def keys_of_index_information(index_information):
    return [x['key'] for x in index_information.values()]


def mg_collection_string(mgc):
    return mgc.database.name + '/' + mgc.name


def imap_with_error_handling(apply_fun, error_fun, except_errors=(Exception,), iterator=None):
    """
    imap_with_error_handling(
    """
    assert iterator is not None, "You're iterator was None!"
    for i, x in enumerate(iterator):
        try:
            yield apply_fun
        except Exception as error:
            error_fun(x=x, error=error, i=i)


def convert_dict_for_mongo(d):
    n = {}
    for k, v in d.items():
        if isinstance(v, dict):
            n[k] = convert_dict_for_mongo(v)
        else:
            if isinstance(k, unicode):
                for i in ['utf-8', 'iso-8859-1']:
                    try:
                        k = k.encode(i)
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        continue
            if isinstance(v, (int64, int32)):
                v = int(v)
            elif isinstance(v, (float64, float32)):
                v = float(v)
            elif isinstance(v, (ndarray, np.matrixlib.defmatrix.matrix)):
                if v.dtype == int32 or v.dtype == int64:
                    v = v.astype(int).tolist()
                elif v.dtype == float32 or v.dtype == float64:
                    v = v.astype(float).tolist()
                else:
                    v = v.tolist()
            elif isinstance(v, unicode):
                for i in ['utf-8', 'iso-8859-1']:
                    try:
                        v = v.encode(i)
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        continue
            elif hasattr(v, 'isoformat'):
                v = v.isoformat()
            n[k] = v
    return n


def iterate_cursor_and_recreate_if_cursor_not_found(cursor_creator, doc_process, start_i=0,
                                                    print_progress_fun=None, on_error=None):
    """
    Iterates cursor, calling doc_process at every step, and recreates the cursor and restarts the loop if
    there's a CursorNotFound error.
    cursor_creator(skip) is a function that returns a cursor that skips skip docs.

    Initially, the cursor is created calling cursor_creator(skip=0), and the loop keeps track of the number of docs
    it processed. If a CursorNotFound error is raised at step i, the cursor is recreated calling
    cursor_creator(skip=i), and the loop is restarted.

    This is useful for situations where the cursor may "timeout" during voluminous processes.

    print_progress_fun(i, doc) (optional) is a function that will be called at every iteration,
    BEFORE doc_process is called, usually to print or log something about the progress.
    """
    while True:
        it = cursor_creator(skip=start_i)
        try:
            for i, doc in enumerate(it, start_i):
                if print_progress_fun is not None:
                    try:
                        print_progress_fun(i, doc)
                    except (CursorNotFound, KeyboardInterrupt, StopIteration) as e:
                        raise e
                    # except Exception as e:
                    #     if on_error is not None:
                    #         on_error(doc=doc, error=e, i=i)
                    #     else:
                    #         raise e
                try:
                    doc_process(doc)
                except Exception as e:
                    if on_error is not None:
                        on_error(obj=doc, error=e, i=i)
                    else:
                        raise e
            break
        except CursorNotFound:
            start_i = i
        except KeyboardInterrupt:
            break
        except StopIteration:
            break


def get_db_and_collection_and_create_if_doesnt_exist(db, collection, mongo_client=None):
    if mongo_client is None:
        mongo_client = MongoClient()

    try:
        mongo_client.create_database(db)
    except Exception as e:
        print(e)

    try:
        mongo_client[db].create_collection(collection)
    except Exception as e:
        print(e)
    return mongo_client[db][collection]


def random_selection_iter_from_cursor(cursor, n_selections):
    total = cursor.count()
    if n_selections >= total:
        return cursor
    else:
        choice_idx = iter(sorted(random.choice(total, n_selections, replace=False)))

        def selection_iterator():
            current_choice_idx = choice_idx.next()
            for i, doc in enumerate(cursor):
                if i == current_choice_idx:
                    yield doc
                    current_choice_idx = choice_idx.next()

        return selection_iterator()


def get_random_doc(collection, *args, **kwargs):
    c = collection.find(*args, **kwargs)
    count = c.count()
    if count == 0:
        raise RuntimeError("No documents with specified conditions were found!")
    else:
        return c.limit(-1).skip(random.randint(0, count)).next()


def mk_find_in_field_logical_query(field, query):
    """
    Allows one to create "find in field" query of any logical complexity (since AND, OR, and NOT are supported).

    field is the field to consider

    query can be a string, a tuple, or a list, and can be nested:
        if query is a string, it is is considered to be "must be equal to this"
            if the string starts with "-", it is considered to be "must NOT equal to this"
        if query is a tuple, take the conjunction (AND) of the tuple's elements
        if query is a list, take the disjunction (OR) of the list's elements
    """
    if isinstance(query, basestring):
        if query[0] == '-':
            return {field: {'$not': {'$eq': query[1:]}}}
        else:
            return {field: query}
    elif isinstance(query, tuple):
        return {'$and': map(lambda q: mk_find_in_field_logical_query(field, q), query)}
    elif isinstance(query, list):
        return {'$or': map(lambda q: mk_find_in_field_logical_query(field, q), query)}
    else:
        raise TypeError("query must be a string, tuple, or list")


def copy_collection_from_remote_to_local(remote_client, remote_db, remote_collection,
                                         local_db=None, local_collection=None,
                                         max_docs_per_collection=inf, verbose=False):
    local_db = local_db or remote_db
    local_collection = local_collection or remote_collection

    remote_collection_connection = remote_client[remote_db][remote_collection]

    local_db_connection = mg.MongoClient()[local_db]
    if local_collection in local_db_connection.collection_names():
        print("Local collection '{}' existed and is being deleted".format(local_collection))
        try:
            local_db_connection[local_collection].drop()
        except mg.errors.OperationFailure as e:
            print("  !!! Nope, can't delete that: {}".format(e.message))
    local_collection_connection = local_db_connection[local_collection]
    for i, d in enumerate(remote_collection_connection.find()):
        if i < max_docs_per_collection:
            if verbose:
                printProgress("item {}".format(i))
            local_collection_connection.insert(d)
        else:
            break


def get_dict_with_key_from_collection(key, collection):
    try:
        return collection.find_one({key: {'$exists': True}}).get(key)
    except AttributeError:
        return None


def insert_df(df, collection, delete_previous_contents=False, **kwargs):
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


def to_df(cursor, roll_out_col=None, rm_id=True):
    # if isinstance(cursor, dict):
    if not isinstance(cursor, list):
        df = pd.DataFrame(list(cursor))
    else:
        df = pd.DataFrame(cursor)
    if rm_id:
        df = rm_cols_if_present(df, ['_id'])
    if roll_out_col:
        # rollout the col
        df = rollout_cols(df, roll_out_col)
        df = pd.concat([rm_cols_if_present(df, roll_out_col), pd.DataFrame(list(df[roll_out_col]))], axis=1)
    return df


def mongorestore(mongodump_file, db, collection, extra_options='', print_the_command=False):
    db, collection = _get_db_and_collection_from_filename(mongodump_file, db=db, collection=collection)
    command = 'mongorestore  --db {db} --collection {collection} {extra_options} {mongodump_file}'.format(
        db=db, collection=collection, extra_options=extra_options, mongodump_file=mongodump_file
    )
    if print_the_command:
        print command
    p = subprocess.Popen(command, shell=True)
    p.wait()  # wait till the process finishes


def backup_to_s3(db, collection, extra_options='', bucket_name=s3_backup_bucket_name, folder=None):
    zip_filename = 'mongo_{db}_{collection}___{date}.bson.gz'.format(db=db, collection=collection,
                                                              date=datetime.now().strftime('%Y-%m-%d-%H%M'))
    extra_options = extra_options + ' --out -'
    command = 'mongodump --db {db} --collection {collection} {extra_options} | gzip > {zip_filename}'.format(
        db=db, collection=collection, extra_options=extra_options, zip_filename=zip_filename
    )

    print command
    p = subprocess.Popen(command, shell=True)
    p.wait()

    print "uploading file to s3://{bucket_name}{folder}/{zip_filename}".format(
            bucket_name=bucket_name, folder=('/' + folder) if folder else '', zip_filename=zip_filename
        )
    s3 = S3(bucket_name=bucket_name)
    s3.dumpf(zip_filename, zip_filename, folder=folder)
    print "removing {zip_filename}".format(zip_filename=zip_filename)
    os.remove(zip_filename)


def restore_from_s3_dump(s3_zip_filename, db=None, collection=None, extra_options='',
                         bucket_name=s3_backup_bucket_name, folder=None, print_the_command=True):

    db, collection = _get_db_and_collection_from_filename(s3_zip_filename, db=db, collection=collection)

    print "copy s3://{bucket_name}{folder}/{zip_filename} to local {zip_filename}".format(
            bucket_name=bucket_name, folder=('/' + folder) if folder else '', zip_filename=s3_zip_filename
    )
    s3 = S3(bucket_name=bucket_name)
    s3.loadf(key_name=s3_zip_filename, local_file_name=s3_zip_filename, folder=folder, bucket_name=bucket_name)

    print "unzip {zip_filename}".format(zip_filename=s3_zip_filename)
    unzipped_filename = s3_zip_filename.replace('.gz', '')
    ungzip(gzip_file=s3_zip_filename, destination_file=unzipped_filename)

    print "removing {gzip_file}".format(gzip_file=s3_zip_filename)
    os.remove(s3_zip_filename)

    mongorestore(mongodump_file=unzipped_filename, db=db, collection=collection,
                 extra_options=extra_options, print_the_command=print_the_command)

    print "removing {unzipped_filename}".format(unzipped_filename=unzipped_filename)
    os.remove(unzipped_filename)


def _get_db_and_collection_from_filename(filename, db=None, collection=None):
    if db is None:
        if collection:
            db_coll_re = re.compile('mongo_(.*?)_{collection}___'.format(collection=collection))
            return db_coll_re.findall(filename)[0], collection
        else:
            db_coll_re = re.compile('mongo_([^_]+)_(.*?)___')
            return db_coll_re.findall(filename)[0]
    else:
        if collection:
            return db, collection
        else:
            db_coll_re = re.compile('mongo_{db}_(.*?)___'.format(db=db))
            return db, db_coll_re.findall(filename)[0]

# def _integrate_filt(filt, *args, **kwargs):
#
#     if len(args) > 0:
#         if 'spec' in kwargs:
#             raise TypeError("got multiple values for keyword argument 'spec' (one in args, one in kwargs")
#         args = list(args)
#         kwargs['spec'] = args.pop(0)
#         args = tuple(args)
#
#     if 'spec' in kwargs:
#         recursively_update_with(kwargs['spec'], filt)
#     else:
#         kwargs['spec'] = filt
#
#     return args, kwargs
#
#
class FilteredCollection(Collection):
    def __init__(self, mgc, filt=None):
        self.mgc = mgc
        if filt is None:
            filt = {}
        self.filt = filt.copy()

    def _integrate_filt(self, *args, **kwargs):

        if len(args) > 0:
            if 'spec' in kwargs:
                raise TypeError("got multiple values for keyword argument 'spec' (one in args, one in kwargs")
            args = list(args)
            kwargs['spec'] = args.pop(0)
            args = tuple(args)

        if 'spec' in kwargs:
            recursively_update_with(kwargs['spec'], self.filt)
        else:
            kwargs['spec'] = self.filt

        return args, kwargs

    def find(self, *args, **kwargs):
        """
        Filtered version of pymongo collection find.
        """
        args, kwargs = self._integrate_filt(*args, **kwargs)
        return self.mgc.find(*args, **kwargs)

    def find_one(self, *args, **kwargs):
        """
        Filtered version of pymongo collection find_one.
        """
        args, kwargs = self._integrate_filt(*args, **kwargs)
        return self.mgc.find_one(*args, **kwargs)

    def count(self):
        """
        Filtered version of pymongo collection count.
        """
        return self.mgc.find(self.filt).count()

    def __getattr__(self, item):
        """
        Forward all other things to self.mgc
        """
        return self.mgc.__getattr__(item)
#
#     # def __getattribute__(self, name):
#     #     attr = super(FilteredCollection, self).__getattribute__(name)
#     #     if hasattr(attr, '__call__'):
#     #
#     #         def newfunc(*args, **kwargs):
#     #             args, kwargs = self._integrate_filt(*args, **kwargs)
#     #             result = attr(*args, **kwargs)
#     #             return result
#     #
#     #         return newfunc
#     #     else:
#     #         return attr
#
#
# def filtered_mgc(self, filt):
#     filt = filt.copy()
#
#     def find(self, *args, **kwargs):
#         args, kwargs = _integrate_filt(filt, *args, **kwargs)
#         return object.__getattribute__(self, 'find')(*args, **kwargs)
#
#     def __getattribute__(self, name):
#         if name == 'find':
#             def newfunc(*args, **kwargs):
#                 args, kwargs = _integrate_filt(filt, *args, **kwargs)
#                 result = object.__getattribute__(self, name)(*args, **kwargs)
#                 return result
#             return newfunc
#         elif name == 'find_one':
#             def newfunc(*args, **kwargs):
#                 args, kwargs = _integrate_filt(filt, *args, **kwargs)
#                 result = object.__getattribute__(self, name)(*args, **kwargs)
#                 return result
#             return newfunc
#         elif name == 'count':
#             def newfunc(*args, **kwargs):
#                 args, kwargs = _integrate_filt(filt, *args, **kwargs)
#                 result = object.__getattribute__(self, name)(*args, **kwargs)
#                 return result
#
#             return newfunc
#         else:
#             return object.__getattribute__(self, name)
#         #
#         # attr = self.__getattr__(name)
#         # if hasattr(attr, '__call__'):
#         #     def newfunc(*args, **kwargs):
#         #         args, kwargs = _integrate_filt(filt, *args, **kwargs)
#         #         result = object.__getattribute__(self, name)(*args, **kwargs)
#         #         return result
#         #
#         #     return newfunc
#         # else:
#         #     return attr
#
#     return inject_method(self, __getattribute__, '__getattribute__')
