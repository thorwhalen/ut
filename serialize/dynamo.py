import os
from time import sleep
import boto.dynamodb2
from .khan_logger import KhanLogger

__author__ = 'mattjmorris'

class Dynamo(object):

    def __init__(self, access_key=None, secret=None):
        """
        If access_key and/or secret are not passed in, assumes we are accessing erenev's aws account and that the
        access info is stored as environment variables on the current server.

        Connection and Table are available to clients via self properties, in case clients wish to use those objects
        directly.
        """
        access_key = access_key or os.getenv('VEN_S3_ACCESS_KEY')
        secret = secret or os.getenv('VEN_S3_SECRET')
        self.connection=boto.dynamodb2.connect_to_region(region_name='eu-west-1', aws_access_key_id=access_key, aws_secret_access_key=secret)
        self.logger = KhanLogger(origin=self.__class__.__name__)

    def modify_throughput(self, requested_read, requested_write, table):
        """
        Used to change the throughput of a specific table
        """
        read, write, num_dec_today, table_status = self.get_table_info(table)

        while requested_read != read or requested_write != write:

            self.logger.info(msg="Modifying {} from {}, {} to {}, {}".format(table.table_name, read, write,
                                                                             requested_read, requested_write))

            new_read, new_write = self._new_read_write(read, requested_read, write, requested_write)

            self.logger.info(msg="going to request read {} and write {}".format(new_read, new_write))

            if (new_read < read or new_write < write) and num_dec_today >= 4:
                # Todo - replace with custom error and handle in client code
                raise ValueError("Sorry, can't do any more decreases today.")
            table.update(throughput={'read': new_read, 'write': new_write})

            sleep_secs = 30
            table_status = 'UPDATING'
            self.logger.info(msg="Sleeping for {} secs before starting".format(sleep_secs))
            sleep(sleep_secs)
            while table_status == 'UPDATING':
                self.logger.info(msg="Sleeping for {} secs".format(sleep_secs))
                sleep(sleep_secs)
                read, write, num_dec_today, table_status = self.get_table_info(table)

        return read, write

    def _new_read_write(self, read, requested_read, write, requested_write):
        """
        Ensures that we change throughput in the correct amounts so as to not cause DDB to yell at us.
        """
        if requested_read == 0:
            read_change_prop = 0
        else:
            read_change_prop = requested_read / float(read)

        # max increase allowed is a doubling
        if read_change_prop > 2:
            new_read = read * 2
        else:
            new_read = requested_read

        if requested_write == 0:
            write_change_prop = 0
        else:
            write_change_prop = requested_write / float(write)

        if write_change_prop > 2:
            new_write = write * 2
        else:
            new_write = requested_write

        return new_read, new_write

    def get_table_info(self, table):
        """
        Returns meta information about the table, such as read speed, write speed, current status,
        and number of decreases today. Useful for figuring out how to change throughput.
        """
        desc = table.describe()
        status = desc['Table']['TableStatus']
        throughput = desc['Table']['ProvisionedThroughput']
        num_decreases = throughput['NumberOfDecreasesToday']
        read = throughput['ReadCapacityUnits']
        write = throughput['WriteCapacityUnits']
        return read, write, num_decreases, status
