
__author__ = 'mattjmorris'

from .dynamo import Dynamo
from boto.dynamodb2.table import Table

class DDBKids(Dynamo):

    @classmethod
    def from_test_mode(cls, access_key=None, secret=None):
        """
        Use this for getting an instance of this class that uses test tables.
        """
        instance = cls(access_key, secret)
        instance.table = Table('test_kids', connection=instance.connection)
        return instance

    def __init__(self, access_key=None, secret=None):
        super(DDBKids, self).__init__(access_key, secret)
        self.table = Table('kids', connection=self.connection)

    def set_max_kid(self, account_name, kid):
        """
        Set the max kid used for an account
        :param account_name:
        :param kid: int >= 0.
        """
        return self.table.put_item({'account_name': account_name, 'kid': kid}, overwrite=True)

    def get_max_kid(self, account_name):
        """
        Get the max kid already used for an account. If the account does not exist, create it in the db with a KID of 0.
        :param account_name:
        """
        res = self.table.get_item(account_name=account_name)
        if res['kid']:
            return int(res['kid'])
        else:
            self.logger.warn("Creating a new (max) KID entry for account {} because it did not yet exist in ddb_kids".format(account_name))
            self.set_max_kid(account_name, 0)
            return 0

    def modify_throughput(self, requested_read, requested_write, table=None):
        table = table or self.table
        return super(DDBKids, self).modify_throughput(requested_read, requested_write, table)

    def truncate_table(self):
        """
        WARNING! Only use for test mode table
        """
        assert self.table.table_name == 'test_kids', "Will only truncate test table. To truncate production table, run code manually"
        with self.table.batch_write() as batch:
            for item in self.table.scan():
                batch.delete_item(dt=item['dt'])