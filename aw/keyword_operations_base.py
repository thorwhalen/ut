import logging
from ut.util.importing import get_environment_variable
from adwords.awq import AWQ
from adwords.connection import Connection
from adwords.gmoney import GMoney
from adwords.job_checker import JobChecker
from adwords.job_results import JobResults
from serialize.khan_logger import KhanLogger
from adwords.mutations import Mutations
from adwords.operations import Operations


class KeywordOperationsBase(object):
    def __init__(self, account_id='7998744469', store=None, min_money=None, max_money=None, logging_level=logging.DEBUG, chk_size=5000):
        """
        Pass in min and max (in Euros, NOT micros) based bid if you want to override the GMoney defaults
        store: storage that acts like an HDF5 store
        """
        self.conn = Connection(password=get_environment_variable('VEN_ADWORDS_PASSWORD'),
                               developer_token=get_environment_variable('VEN_ADWORDS_TOKEN'),
                               account_id=account_id)
        self.awq = AWQ(self.conn)
        self.gmoney = GMoney(min_money=min_money, max_money=max_money)
        self.ops = Operations(self.gmoney)
        self.mutations = Mutations(self.conn)

        self.chk_size = chk_size

        self.store = store

        self.logger = KhanLogger(level=logging_level, origin=self.__class__.__name__)

        # convenience properties to make it easier to call methods in succession without tracking job ids
        self.job_ids = None
        self.job_ids_completed = None
        self.job_ids_failed = None

    def _contains(self, list1, list2):
        """
        Returns true if list1 contains list2, otherwise returns false
        """
        if set(list1) - set(list2):
            return False
        else:
            return True

    # TODO - create another method that allows pinging to see if all done, so can cycle through accounts
    def get_completed_and_failed_jobs(self, job_ids=None):
        """
        Don't pass in job ids if you want to check on all job ids resulting from the last set of operations
        """
        job_ids = job_ids or self.job_ids
        jc = JobChecker(adwords_connection=self.conn, job_ids=job_ids)

        jc.poll_until_complete()
        self.job_ids_completed =jc.job_ids_completed
        self.job_ids_failed = jc.job_ids_failed

        #if self.job_ids_failed:
        #    self.logger.structured_log(level=logging.WARN, origin='keyword operations', msg="Failed jobs: {}".format(self.job_ids_failed))

        return self.job_ids_completed, self.job_ids_failed

    def get_errors(self, job_ids=None):
        """
        Don't pass in job_ids list if you want to get errors for all completed jobs
        """
        job_ids = job_ids or self.job_ids_completed
        jr = JobResults(adwords_connection=self.conn, job_ids=job_ids, logging_level=logging.ERROR)
        errors_by_job = jr.get_errors()
        for job_id in list(errors_by_job.keys()):
            locations = [t[0] for t in errors_by_job[job_id]]
            errors = [t[1] for t in errors_by_job[job_id]]
            df = self.store['jobs/_' + job_id.replace('-', 'N')]
            error_df = df.iloc[locations]
            error_df['error']=errors
            self.store.put('failed_jobs/_' + job_id.replace('-', 'N'), error_df)
        # return all of the job_ids that had errors
        return list(errors_by_job.keys())