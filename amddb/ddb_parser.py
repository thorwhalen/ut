_author__ = 'mattjmorris'

from boto.dynamodb2.table import Table
import boto.dynamodb2
from ut.util.importing import get_environment_variable
from datetime import datetime
from khan_utils.encoding import to_unicode_or_bust, to_utf8_or_bust


# TODO ! put in rate-limiting code here for writes (if boto not handling this correctly). Also, handle reads > 1 Meg here
class DDBParser(object):

    def __init__(self,
                 access_key=None,
                 secret=None,
                 failed_table_name='parsed_simple_failed',
                 missing_table_name='parsed_simple_missing',
                 success_table_name='parsed_simple_good'
                 ):
        """
        If access_key and/or secret are not passed in, assumes we are accessing erenev's aws account and that the
        access info is stored as environment variables on the current server.

        Connection and Table are available to clients via self properties, in case clients wish to use those objects
        directly.
        """

        access_key = access_key or get_environment_variable('VEN_S3_ACCESS_KEY')
        secret = secret or get_environment_variable('VEN_S3_SECRET')
        self.connection=boto.dynamodb2.connect_to_region(region_name='eu-west-1', aws_access_key_id=access_key, aws_secret_access_key=secret)

        self.failed_table = Table(failed_table_name, connection=self.connection)
        self.missing_table = Table(missing_table_name, connection=self.connection)
        self.success_table = Table(success_table_name, connection=self.connection)

    def save_results(self, searchterm, col1, col2, col3):
        self.success_table.put_item(data={'searchterm': searchterm, })

    def save_failure(self, searchterm, msg):
        self.failed_table.put_item(data={'searchterm': searchterm, 'msg': msg})

    def save_missing(self, searchterm=None, searchterms=None):
        # assert searchterm != searchterms, "You must either pass in a searchterm or a list of searchterms, but not both"
        # if searchterm:
        self.missing_table.put_item(data={'searchterm': searchterm}, overwrite=True)
        # else:
        #     with self.missing_table.batch_write() as batch:
        #         for s in searchterms:
        #             batch.put_item(data=s, overwrite=True)


    def get_slurp_info(self, search_term_=None):
        """
        search_term_ can be either a string or a list of strings. Each string should be a search term you are looking
        for in the db.
        Returns either a single list of key-value tuples (if search_term_ was a string)
        or a list of key-value tuples (if search_term_ was a list)
        Each list of key-value tuples can easily be converted to a dict or an OrderedDict by the client.
        """

        # TODO! make the search_term list a set so don't get duplicate error from dynamo

        # searchterm_ is a STRING
        if isinstance(search_term_, basestring):
            if search_term_:
                slurp_info = (self.slurps_table.get_item(searchterm=search_term_)).items()
            else:
                slurp_info = []

        # searchterm is a LIST of strings
        elif isinstance(search_term_, list):
            if search_term_:
                list_of_searchterms = [{'searchterm': st} for st in search_term_ if st]
                res = self.slurps_table.batch_get(list_of_searchterms)
                try:
                    slurp_info = [i.items() for i in res]
                except (StopIteration, IndexError):
                    # If res is empty, we get one of these errors when trying to iterate.
                    slurp_info = []
            else:
                slurp_info = []

        # searchterm is an unexpected type
        else:
            raise TypeError, "search_term_ must be a dict or a list of dicts, not a {}".format(type(search_term_))

        return slurp_info

    def existing_and_missing(self, searchterm_list):
        """
        Takes a list of searchterm strings and returns a list of searchterm strings that were found in the db (in unicode)
        and a list of the searchterms that were missing from the found results
        """
        # make sure in utf8 before we send request to the db
        input_sts_utf8 = [to_utf8_or_bust(i) for i in searchterm_list]
        found_sts_info = self.get_slurp_info(input_sts_utf8)
        found_sts_uni = [to_unicode_or_bust(dict(i)['searchterm']) for i in found_sts_info]
        input_sts_uni = [to_unicode_or_bust(i) for i in input_sts_utf8]
        missing_sts_uni = list(set(input_sts_uni) - set(found_sts_uni))
        # make sure to compare unicode to unicode
        return found_sts_uni, missing_sts_uni

    def get_table(self, table_name):
        """
        Convenience method for client who may wish to get a specific table from the DynamoDB connection
        """
        return Table(table_name, connection=self.connection)

    def truncate_slurp_table(self):
        """
        WARNING! Only use for test mode table
        """
        assert self.test_mode==True, "Will only truncate test slurps table. To truncate production table, run code manually"
        with self.slurps_table.batch_write() as batch:
            for item in self.slurps_table.scan():
                batch.delete_item(searchterm=item['searchterm'])

    def truncate_failed_slurp_table(self):
        """
        WARNING! Only use for test mode table
        """
        assert self.test_mode==True, "Will only truncate test failed slurps table. To truncate production table, run code manually"
        with self.failed_slurps_table.batch_write() as batch:
            for item in self.failed_slurps_table.scan():
                batch.delete_item(searchterm=item['searchterm'], datetime=item['datetime'])
