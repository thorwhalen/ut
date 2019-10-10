
__author__ = 'mattjmorris'

from .dynamo import Dynamo
from boto.dynamodb2.table import Table
from datetime import datetime, date, timedelta
from pandas import DataFrame
import re

DAY_STR_FORMAT = "%Y-%m-%d"
DAY_STR_RE = re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
SECOND_STR_FORMAT = "%Y-%m-%d %H:%M:%S"
SECOND_STR_RE = re.compile(r'^(\d{4})-(\d{2})-(\d{2})\s(\d{2}:\d{2}:\d{2})$')

class DDBRuns(Dynamo):

    @classmethod
    def from_test_mode(cls, access_key=None, secret=None):
        """
        Use this for getting an instance of this class that uses test tables.
        """
        instance = cls(access_key, secret)
        instance.table = Table('test_runs', connection=instance.connection)
        return instance

    def __init__(self, access_key=None, secret=None):
        """
        When called directly (as should be done for production code), sets table to the production 'runs' table.
        """
        super(DDBRuns, self).__init__(access_key, secret)
        self.table = Table('runs', connection=self.connection)

    def save_new_run(self, dt_str=None, start_date_str=None, end_date_str=None):
        """
        dt_str = datetime of run. Defaults to now.
        start_date_str = the start date for look-back of query performance data processing. * No default
        end_date_str = the end date for query performance data processing. Defaults to today.
        """
        assert start_date_str, "start_date_str is required when saving a new run to runs table."
        assert DAY_STR_RE.match(start_date_str)
        if end_date_str:
            assert DAY_STR_RE.match(end_date_str)
        if dt_str:
            assert SECOND_STR_RE.match(dt_str)

        dt_str = dt_str or datetime.now().strftime(SECOND_STR_FORMAT)
        end_date_str = end_date_str or datetime.now().strftime(DAY_STR_FORMAT)
        return self.table.put_item(data={'dt': dt_str, 'start': start_date_str, 'end': end_date_str})

    def most_recent_start_date_str(self):
        """
        :return: a string representing most recent start date from db
        """
        df = self.get_runs_df()
        if df.empty:
            return None
        else:
            # should already be sorted, but just in case...
            df.sort(columns=['dt'], ascending=True, inplace=True)
            return df.iloc[len(df)-1]['start']

    def most_recent_end_date_str(self):
        """
        :return: a string representing most recent end date from db
        """
        df = self.get_runs_df()
        if df.empty:
            return None
        else:
            # should already be sorted, but just in case...
            df.sort(columns=['dt'], ascending=True, inplace=True)
            return df.iloc[len(df)-1]['end']

    def get_runs_df(self):
        """
        Returns all table as dataframe, sorted with most recent entry on bottom (ascending order)
        """
        df = DataFrame([{k: v for k, v in list(r.items())} for r in self.table.scan()])
        if df.empty:
            return df
        else:
            df.sort(columns=['dt'], ascending=True, inplace=True)
            # force df to have columns in this order
            return df[['dt', 'start', 'end']]

    def modify_throughput(self, requested_read, requested_write, table=None):
        table = table or self.table
        return super(DDBRuns, self).modify_throughput(requested_read, requested_write, table)

    def truncate_table(self):
        """
        WARNING! Only use for test mode table
        """
        assert self.table.table_name == 'test_runs', "Will only truncate test table. To truncate production table, run code manually"
        with self.table.batch_write() as batch:
            for item in self.table.scan():
                batch.delete_item(dt=item['dt'])


    def thors_start_end_date_strings(self, new_run=True, days_ago_start=30):
        if new_run:
            if days_ago_start is not None:
                print(days_ago_start)
                start_date_str = self._days_ago_str(days_ago_start)
            else:
                start_date_str = self.most_recent_end_date_str()
            end_date_str = date.today().strftime(DAY_STR_FORMAT)
        else:
            start_date_str = self.most_recent_start_date_str()
            end_date_str = self.most_recent_end_date_str()
            assert start_date_str, "Start date string is None, please check the database since we are not doing a new run"
            assert end_date_str, "End date string is None, please check the database since we are not doing a new run"
        return start_date_str, end_date_str

    def _days_ago_str(self, num_days_ago):
        return (date.today() - timedelta(days=num_days_ago)).strftime(DAY_STR_FORMAT)

    def start_end_date_strings(self, new_run=True, days_ago_start=30):
        if new_run:
            start_date_str = self.most_recent_end_date_str() or self._days_ago_str(days_ago_start)
            end_date_str = date.today().strftime(DAY_STR_FORMAT)
        else:
            start_date_str = self.most_recent_start_date_str()
            end_date_str = self.most_recent_end_date_str()
        return start_date_str, end_date_str
