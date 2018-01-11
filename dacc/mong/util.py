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
from pymongo.collection import Collection
from pymongo.errors import InvalidOperation

from ut.daf.to import dict_list_of_rows
from ut.daf.manip import rm_cols_if_present
from ut.daf.manip import rollout_cols
from ut.serialize.s3 import S3
from ut.pfile.to import ungzip
from ut.util.log import printProgress
import numpy as np
from ut.pdict.manip import recursively_update_with
from ut.dpat.buffer import ThreshBuffer

s3_backup_bucket_name = 'mongo-db-bak'


def autorefresh_cursor_iterator(cursor):
    def refresh_when_cursor_not_found(cursor):
        while True:
            try:
                yield cursor.next()
            except CursorNotFound:
                cursor = restart_find_cursor(cursor)
                yield cursor.next()

    return refresh_when_cursor_not_found(cursor)


def restart_find_cursor(cursor, docs_retrieved_so_far=None):
    if docs_retrieved_so_far is None:
        docs_retrieved_so_far = cursor._Cursor__retrieved
    kwargs = dict(spec=cursor._Cursor__query_spec(),
                  fields=cursor._Cursor__fields,
                  skip=cursor._Cursor__skip + docs_retrieved_so_far)
    if cursor._Cursor__limit:
        new_limit = max(0, cursor._Cursor__limit - docs_retrieved_so_far)
        if new_limit > 0:
            kwargs = dict(kwargs, limit=new_limit)
        else:  # in this particular case, we should return an empty cursor. This is my hack for that.
            new_cursor = cursor._Cursor__collection.find(**kwargs).limit(1)
            new_cursor.next()
            return new_cursor

    return cursor._Cursor__collection.find(**kwargs)


class BulkMgUpdates(object):
    """
    A class to accumulate update instructions and flush them (to perform a bulk mongo update).
    See also BulkUpdateBuffer.

    When constructed, once must specify a collection object mgc that the data must be stored in.

    The push method pushes a (spec, document) tuple to the bulk_op list.
    The flush method actually does the bulk write of the items pushed so far.

    These methods can be used "manually" to get data written, but a common use of BulkMgUpdates is to used the feed
    method, which takes an iterator of items, pushes the corresponding (spec, document) tuples, flushes at regular
    intervals, and when the iterator is completely consumed, will do a final flush.

    The regularity of the flushes during a feed call is controlled by the flush_every attribute.
    How to get (spec, document) tuples from the items the iterator yields is specified by the get_spec_and_doc
    attribute: A callable taking an item the iterator yields and returning the corresponding (spec, document)
    """

    def __init__(self, mgc, flush_every=500, get_spec_and_doc=None):
        """

        :param mgc: mongo collection (pymongo.collection.Collection object)
        :param flush_every: How often should data be flushed to the db when calling the feed method
        :param get_spec_and_doc: The function to be applied to the elements of the iterator in the feed method to get
            (spec, document) tuples from iterator elements. If not specified, will assume the iterator is feeding
            (spec, document) tuples.
        """
        self.mgc = mgc
        self.updater = None
        self.initialize()
        self.flush_every = flush_every
        if get_spec_and_doc is None:
            get_spec_and_doc = lambda x: x
        self.get_spec_and_doc = get_spec_and_doc

    def initialize(self):
        self.updater = self.mgc.initialize_unordered_bulk_op()

    def push(self, spec, document):
        return self.updater.find(spec).upsert().update(document)

    def flush(self):
        r = self.updater.execute()
        self.initialize()
        return r

    def feed(self, it):
        self.initialize()
        for i, x in enumerate(it, 1):
            spec, document = self.get_spec_and_doc(x)
            self.push(spec, document)
            if i % self.flush_every == 0:
                self.flush()
        self.flush()


class BulkUpdateBuffer(ThreshBuffer):
    def __init__(self, mgc, max_buf_size=500, upsert=False):
        """
        Accumulate and execute bulk update operations.

        An object to have the convenience to just "push and forget" bulk operations.
        By push, we mean "just add an update operation".
        By forget, we mean "don't have to remember to execute the bulk operation since the object will do so
        automatically as soon as the threshold flush_when_buffer_size_is_over is reached.

        Note: That said, you can't completely forget since you should run self.flush_operations() at the end
         of an iteration of add_operation instructions to make sure the remainder of operations (accumulated, but
         below the threshold) will be written to the target collection.

        :param mgc: The mongo collection (a pymongo.collection.Collection object) to update
        :param flush_when_buffer_size_is_over: The number of operations after which to do a bulk update.
        :param upsert: Whether to use update with upsert or not (default False)

        >>> from pymongo import MongoClient
        >>> from numpy.random import randint
        >>>
        >>> i = 0
        >>> def rand_doc():
        ...     global i
        ...     i += 1
        ...     return {'a': i, 'b': list(randint(0, 9, 3))}
        ...
        >>> # get an empty test collection
        >>> mgc = MongoClient()['test']['test']
        >>> _ = mgc.remove({})
        >>> print(len(list(mgc.find())))
        0
        >>> # Use BulkUpdateBuffer to bulk insert 7 docs (with flush when buffer is size 3)
        >>> bub = BulkUpdateBuffer(mgc, max_buf_size=3, upsert=True)
        >>> for i in range(7):
        ...     doc = rand_doc()
        ...     _ = bub.push({'spec': {'a': doc['a']},
        ...                        'document': {'$set': {'b': doc['b']}}});
        >>> # Note that 6 out of 7 docs are in the collection
        >>> print(len(list(mgc.find())))
        6
        >>> # This is why, when one is done with consuming the doc iterator, one should always flush the operations.
        >>> _ = bub.flush();
        >>> print(len(list(mgc.find())))
        7
        >>> # showing how, when using iterate, all docs are flushed
        >>> it = ({'spec': {'a': doc['a']},
        ...        'document': {'$set': {'b': doc['b']}}}
        ...       for doc in (rand_doc() for i in range(4)))
        >>> _ = bub.iterate(it)
        >>> print(len(list(mgc.find())))
        11
        """
        self.mgc = mgc
        super(BulkUpdateBuffer, self).__init__(thresh=max_buf_size)
        self.initialize()
        self.upsert = upsert

    def initialize(self):
        super(BulkUpdateBuffer, self).initialize()
        self._buf_size = 0
        self._buf = self.mgc.initialize_unordered_bulk_op()

    def buf_val_for_thresh(self):
        return self._buf_size

    def _push(self, item):
        if isinstance(item, dict):
            if self.upsert:
                self._buf.find(item.get('spec')).upsert().update(item.get('document'))
            else:
                self._buf.find(item.get('spec')).update(item.get('document'))
            self._buf_size += 1
        else:
            for _item in item:
                self._push(_item)

    def _flush(self):
        try:
            return self._buf.execute()
        except InvalidOperation as e:
            if str(e) != 'No operations to execute':
                raise e

    def push(self, item):
        """
        Push an operation to the buffer
        :param item: A bulk operation. A dict containing a "spec" and a "document" field.
        :return: If operations were flushed, returns what every flush_operation returns, if not returns None.
        """
        return super(BulkUpdateBuffer, self).push(item)

    def flush(self):
        """
        Call bulk_mgc.execute() to write (and flush) all operations that have been added.
        Also reinitialize the buffer_size to 0 and reintialize unordered_bulk_op.
        :return: What ever bulk_mgc.execute() returns
        """
        return super(BulkUpdateBuffer, self).flush()


class KeyedBulkUpdateBuffer(BulkUpdateBuffer):
    def __init__(self, key_fields, mgc, max_buf_size=500, upsert=False, assert_all_keys=True):
        """
        Accumulate and execute bulk update operations with specified key_fields.

        This uses the parent class BulkUpdateBuffer, where items are no longer explicit {'spec': SPEC, 'document': DOC}
        specifications, but "flat docs" from which the key_fields are extracted to form the SPEC dict.

        See BulkUpdateBuffer for mor information.

        :param key_fields: A list/tuple/array/set of fields to use for the 'spec' of an update operation
        :param mgc: The mongo collection (a pymongo.collection.Collection object) to update
        :param flush_when_buffer_size_is_over: The number of operations after which to do a bulk update.
        :param upsert: Whether to use update with upsert or not (default False)
        :param assert_all_keys: Whether to assert that all "spec" keys are present

        >>> from pymongo import MongoClient
        >>> from numpy.random import randint
        >>>
        >>> i = 0
        >>> def rand_doc():
        ...     global i
        ...     i += 1
        ...     return {'a': i, 'b': list(randint(0, 9, 3))}
        ...
        >>> # get an empty test collection
        >>> mgc = MongoClient()['test']['test']
        >>> _ = mgc.remove({})
        >>> print(len(list(mgc.find())))
        0
        >>> # Use BulkUpdateBuffer to bulk insert 7 docs (with flush when buffer is size 3)
        >>> bub = KeyedBulkUpdateBuffer(['a'], mgc, max_buf_size=3, upsert=True)
        >>> for i in range(7):
        ...     _ = bub.push(rand_doc())
        >>> # Note that 6 out of 7 docs are in the collection
        >>> print(len(list(mgc.find())))
        6
        >>> # This is why, when one is done with consuming the doc iterator, one should always flush the operations.
        >>> _ = bub.flush();
        >>> print(len(list(mgc.find())))
        7
        >>> # showing how, when using iterate, all docs are flushed
        >>> it = (rand_doc() for i in range(4))
        >>> _ = bub.iterate(it)
        >>> print(len(list(mgc.find())))
        11
        """
        super(KeyedBulkUpdateBuffer, self).__init__(mgc=mgc, max_buf_size=max_buf_size, upsert=upsert)
        self.key_fields = set(key_fields)
        self.assert_all_keys = assert_all_keys

    def _push(self, item):
        _item = {'spec': {}, 'document': {"$set": {}}}
        for k, v in item.iteritems():
            if k in self.key_fields:
                _item['spec'][k] = v
            else:
                _item['document']['$set'][k] = v
        if self.assert_all_keys:
            assert self.key_fields == set(_item['spec']), \
                "Some update keys are missing. All items should have fields: {}".format(self.key_fields)
        super(KeyedBulkUpdateBuffer, self)._push(_item)


def bulk_update_collection(mgc, operations, verbose=0):
    """
    Has the effect of doing:
    for op in operations:
        mgc.update(spec=op['spec'], document=op['document'])
    but uses bulk operations to do it faster.

    :param mgc: the mongo collection to update
    :param operations: a list of {spec: spec, document: document} dicts defining the updates.
    :param verbose: whether to print info before and after the bulk update
    :return: None
    """
    bulk_mgc = mgc.initialize_unordered_bulk_op()

    for operation in operations:
        bulk_mgc.find(operation.get('spec')).update(operation.get('document'))

    if verbose > 0:
        print('Starting bulk update {}'.format(datetime.now()))

    result = bulk_mgc.execute()

    if verbose > 0:
        print('Stoping bulk update {}'.format(datetime.now()))
        print('Update result {}'.format(result))


def bulk_insert_collection(mgc, docs):
    """
    Has the effect of doing:
    for doc in docs:
        mgc.insert(docs)
    but uses bulk operations to do it faster.

    :param mgc: the mongo collection to update
    :param docs: a list of docs to insert
    :return: what ever bulk_mgc.execute() returns
    """
    bulk_mgc = mgc.initialize_unordered_bulk_op()

    for doc in docs:
        bulk_mgc.insert(doc)

    return bulk_mgc.execute()


def copy_missing_indices_from(source_mgc, target_mgc):
    source_index = source_mgc.index_information()
    target_index = target_mgc.index_information()

    for k, v in source_index.iteritems():
        if k not in target_index:
            cumul = []
            for field, val in v['key']:
                if isinstance(val, float):
                    cumul.append((field, int(val)))
                else:
                    cumul.append((field, val))
            target_mgc.create_index(cumul)


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
