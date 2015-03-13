__author__ = 'thorwhalen'


import pymongo as mg
import pandas as pd
from ut.util.imports.ipython_utils import PPR
from ut.daf.to import dict_list_of_rows
from ut.daf.manip import rm_cols_if_present
from ut.daf.ch import to_utf8


def mdb_info(mg_element=None):
    if mdb_info is None:
        return mdb_info(mg.MongoClient())
    else:
        if isinstance(mg_element, mg.MongoClient):
            return {dbname: mdb_info(getattr(mg_element, dbname)) for dbname in mg_element.database_names()}
        elif isinstance(mg_element, mg.database.Database):
            return {coll_name: getattr(mg_element, coll_name).count() for coll_name in mg_element.collection_names()}


def get_db(db_name='test-database'):
    import pymongo as mg
    connection = mg.MongoClient()
    db = connection[db_name]
    return db


class MongoStruct:
    def __init__(self, obj=None):
        """
        MongoStruct() assigns MongoClient() to .obj
        MongoStruct(mongo_client) assigns the mongo_client to .obj
        MongoStruct(database) assigns the database to .obj
        MongStruct(database_name) assigns the
        """
        self.obj = obj or mg.MongoClient()
        # if isinstance(self.obj, mg.MongoClient):
        #     for dbname in self.obj.database_names():
        #         setattr(self, dbname, MongoStruct(self.obj[dbname]))
        # elif isinstance(self.obj, mg.database.Database):
        #     for coll_name in self.obj.collection_names():
        #         setattr(self, coll_name, self.obj[coll_name])
        if isinstance(self.obj, basestring):
            self.obj = getattr(mg.MongoClient(), self.obj)

        self.refresh()

    def __getitem__(self, val):
        return self.__dict__[val]

    def __str__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.iteritems()))

    def __repr__(self):
        return PPR.format_str(mdb_info(self.obj))

    def refresh(self):
        if isinstance(self.obj, mg.MongoClient):
            for dbname in self.obj.database_names():
                setattr(self, dbname, MongoStruct(self.obj[dbname]))
        elif isinstance(self.obj, mg.database.Database):
            for coll_name in self.obj.collection_names():
                setattr(self, coll_name, self.obj[coll_name])
        # elif isinstance(self.obj, mg.collection.Collection):
        #     for coll_name in self.obj.collection_names():
        #         setattr(self, coll_name, self.obj[coll_name])

    def create_collection_ignore_if_exists(self, collection_name):
        if not isinstance(self.obj, mg.database.Database):
            raise ValueError("self.obj must be a database to do that!")
        try:
            self.obj.create_collection(collection_name)
            self.refresh()
        except Exception:
            pass

    def recreate_collection(self, collection_name):
        if not isinstance(self.obj, mg.database.Database):
            raise ValueError("self.obj must be a database to do that!")
        try:
            self.obj.drop_collection(collection_name)
        except Exception:
            pass
        try:
            self.obj.create_collection(collection_name)
        except Exception:
            pass
        self.refresh()

    @staticmethod
    def get_dict_with_key_from_collection(key, collection):
        try:
            return collection.find_one({key: {'$exists': True}}).get(key)
        except AttributeError:
            return None

    @staticmethod
    def insert_df(df, collection, delete_previous_contents=False, dropna=False, **kwargs):
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
        kwargs = dict(kwargs, **{'w': 0})  # default is w=0 (no replicas)
        if kwargs.get('to_utf8'):
            to_utf8(df, columns=df.columns, inplace=True)
        collection.insert(dict_list_of_rows(df, dropna=dropna), **kwargs)

    @staticmethod
    def to_df(cursor):
        return rm_cols_if_present(pd.DataFrame(list(cursor)), ['_id'])

