
__author__ = 'mattjmorris'

from .dynamo import Dynamo
from boto.dynamodb2.table import Table
from datetime import datetime
from khan_utils.encoding import to_unicode_or_bust, to_utf8_or_bust
from ut.coll import order_conserving


class DDBSlurps(Dynamo):

    @classmethod
    def from_test_mode(cls, access_key=None, secret=None):
        """
        Use this for getting an instance of this class that uses test tables.
        """
        instance = cls(access_key, secret)
        instance.slurps_table = Table('test_slurps', connection=instance.connection)
        instance.failed_slurps_table = Table('test_failed_slurps', connection=instance.connection)
        return instance

    def __init__(self, access_key=None, secret=None):
        """
        ! Use test_mode factory method for instantiating this class with test_slurps and test_failed_slurps tables
        """
        super(DDBSlurps, self).__init__(access_key, secret)

        self.slurps_table = Table('slurps', connection=self.connection)
        self.failed_slurps_table = Table('failed_slurps', connection=self.connection)

    def save_slurp_info(self, slurp_info_, overwrite=True):
        """
        slurp_info_ can either be in the form of a list of dicts or else a single dict.
        If slurp_info is a list, batch write will be used
        """
        if isinstance(slurp_info_, dict):
            self.slurps_table.put_item(slurp_info_, overwrite=overwrite)
        elif isinstance(slurp_info_, list):
            with self.slurps_table.batch_write() as batch:
                for s in slurp_info_:
                    batch.put_item(data=s, overwrite=overwrite)
        else:
            raise TypeError("slurp_info must be a dict or a list of dicts, not a {}".format(type(slurp_info_)))

    def save_failed_slurp(self, searchterm):
        self.failed_slurps_table.put_item(data={'searchterm': searchterm, 'datetime': datetime.now().isoformat()},
                                          overwrite=True)

    def get_slurp_info(self, search_term_=None):
        """
        search_term_ can be either a string or a list of strings. Each string should be a search term you are looking
        for in the db.
        Returns either a single list of key-value tuples (if search_term_ was a string)
        or a list of key-value tuples (if search_term_ was a list)
        Each list of key-value tuples can easily be converted to a dict or an OrderedDict by the client.
        """

        # searchterm_ is a STRING
        if isinstance(search_term_, str):
            if search_term_:
                slurp_info = list((self.slurps_table.get_item(searchterm=search_term_)).items())
            else:
                slurp_info = []

        # searchterm is a LIST of strings
        elif isinstance(search_term_, list):
            if search_term_:
                # create a set of non-empty searchterms. We us a set to avoid a duplicate query error from the db
                set_of_sts = {st for st in search_term_ if st}
                # create a list of dicts from the set
                list_of_st_dicts = [{'searchterm': st} for st in set_of_sts]
                res = self.slurps_table.batch_get(list_of_st_dicts)
                try:
                    slurp_info = [list(i.items()) for i in res]
                except (StopIteration, IndexError):
                    # If res is empty, we get one of these errors when trying to iterate.
                    slurp_info = []
            else:
                slurp_info = []

        # searchterm is an unexpected type
        else:
            raise TypeError("search_term_ must be a dict or a list of dicts, not a {}".format(type(search_term_)))

        return slurp_info

    def existing_and_missing_uni(self, searchterm_list):
        """
        Takes a list of searchterm strings and returns a list of searchterm strings that were found in the db (in unicode)
        and a list of the searchterms that were missing from the found results
        """
        # make sure in utf8 before we send request to the db
        input_sts_utf8 = [to_utf8_or_bust(i) for i in searchterm_list]
        found_sts_info = self.get_slurp_info(input_sts_utf8)
        found_sts_uni = [to_unicode_or_bust(dict(i)['searchterm']) for i in found_sts_info]
        input_sts_uni = [to_unicode_or_bust(i) for i in input_sts_utf8]
        missing_sts_uni = order_conserving.setdiff(input_sts_uni, found_sts_uni)
        return found_sts_uni, missing_sts_uni

    def get_table(self, table_name):
        """
        Convenience method for client who may wish to get a specific table from the DynamoDB connection
        """
        return Table(table_name, connection=self.connection)

    def truncate_failed_slurp_table(self):
        """
        """
        with self.failed_slurps_table.batch_write() as batch:
            for item in self.failed_slurps_table.scan():
                batch.delete_item(searchterm=item['searchterm'])

    def truncate_slurp_table(self):
        """
        WARNING! Only use for test mode table
        """
        assert self.slurps_table.table_name == 'test_slurps', "Will only truncate test slurps table. To truncate production table, run code manually"
        test_slurps_table = Table('test_slurps', connection=self.connection)
        with test_slurps_table.batch_write() as batch:
            for item in self.slurps_table.scan():
                batch.delete_item(searchterm=item['searchterm'])

    def modify_failed_slurps_throughput(self, requested_read, requested_write):
        return self.modify_throughput(requested_read, requested_write, self.failed_slurps_table)

    def modify_slurps_throughput(self, requested_read, requested_write):
        return self.modify_throughput(requested_read, requested_write, self.slurps_table)

    def get_slurps_table_info(self):
        return self.get_table_info(self.slurps_table)

    def get_failed_slurps_table_info(self):
        return self.get_table_info(self.failed_slurps_table)

