__author__ = 'thor'

import pymongo as mg
import os
import re
import pandas as pd
from numpy import inf
import subprocess
from datetime import datetime

from ut.daf.to import dict_list_of_rows
from ut.daf.manip import rm_cols_if_present
from ut.daf.manip import rollout_cols
from ut.serialize.s3 import S3
from ut.pfile.to import ungzip
from ut.util.log import printProgress

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


def copy_collection_from_remote_to_local(remote_client, remote_db, remote_collection,
                                         local_db=None, local_collection=None,
                                         max_docs_per_collection=inf, verbose=False):
    local_db = local_db or remote_db
    local_collection = local_collection or remote_collection

    remote_collection_connection = remote_client[remote_db][remote_collection]

    local_db_connection = mg.MongoClient().get_database(local_db)
    if local_collection in local_db_connection.collection_names():
        print("Local collection '{}' existed and is being deleted".format(local_collection))
        try:
            local_db_connection[local_collection].drop()
        except mg.errors.OperationFailure as e:
            print("  !!! Nope, can't delete that: {}".format(e.message))
    local_collection_connection = local_db_connection.get_collection(local_collection)
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
