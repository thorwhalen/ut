
__author__ = 'mattjmorris'

from .dynamo import Dynamo
from boto.dynamodb2.table import Table
from pandas import DataFrame
from datetime import datetime
from khan_utils.encoding import to_unicode_or_bust, to_utf8_or_bust
from ut.coll import order_conserving


class DDBToBeSlurped(Dynamo):

    def __init__(self, access_key=None, secret=None):
        """
        ! Use test_mode factory method for instantiating this class with test_slurps and test_failed_slurps tables
        """
        super(DDBToBeSlurped, self).__init__(access_key, secret)

        self.table = Table('to_be_slurped', connection=self.connection)

    def save_info(self, search_terms):
        """
        search_terms can either be in the form of a list of dicts or else a single dict.
        If slurp_info is a list, batch write will be used
        """
        if isinstance(search_terms, str):
            search_terms = [search_terms]
        # search_terms = {'searchterm': search_terms}
        search_terms = [{'searchterm': x} for x in search_terms]
        # print search_terms
        with self.table.batch_write() as batch:
            for s in search_terms:
                batch.put_item(data=s, overwrite=True)

    def get_table(self, table_name=None):
        """
        Convenience method for client who may wish to get a specific table from the DynamoDB connection
        """
        table_name = table_name or self.table.table_name
        return Table(table_name, connection=self.connection)

    def truncate_table(self):
        """
        Delete whole table
        """
        with self.table.batch_write() as batch:
            for item in self.table.scan():
                batch.delete_item(searchterm=item['searchterm'])

    def modify_slurps_throughput(self, requested_read, requested_write):
        return self.modify_throughput(requested_read, requested_write, self.table)

    def get_slurps_table_info(self):
        return self.get_table_info(self.table)

    def get_table_as_df(self):
        return DataFrame([dict(r) for r in self.table.scan()])


