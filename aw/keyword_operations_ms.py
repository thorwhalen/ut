from aw.keyword_operations_base import KeywordOperationsBase

__author__ = 'mattjmorris'

import json
import daf.get as daf_get
import logging
from time import sleep


class KeywordOperationsMS(KeywordOperationsBase):

    def modify_bids(self, bid_modifier_func=lambda x: x, kw_df=None, adgroup_ids=None):
        """
        In dataframe format, gets all of the keywords in the current account or set of ad groups within the current
        account (if a list of ad groups is passed in), modifies all of the bids by using the bid_modifier_func, stores
        the resulting dataframes in local storage (for purposes of tracking), and pushes the modifications up to
        Google via bulk mutate.

        Inputs:
        bid_modifier_func: bid modification function, should operate on just the bid value.
        kw_df = pass in a dataframe of the keywords you want modified, or none to force a lookup. If you pass it in,
          must have column names that match those expected by the Operations class.
        adgroup_ids: If you pass in your own kw_df, leave this as None. Otherwise, pass in a list of adgroup ids, or
          none to modify the entire account
        """
        # We reset here so we don't accidentally append to old jobs
        self.job_ids = []

        kw_df = kw_df or self.awq.get_positive_keywords(adgroup_ids)
        # sorting the keywords by adgroup helps speed things up when we later push changes to google
        kw_df.sort(columns=['ad_group_id'], inplace=True)

        kw_df.max_cpc = kw_df.max_cpc.apply(bid_modifier_func)

        kw_df['ops'] = kw_df.apply(self.ops.update_keyword_df_row, axis=1)

        big_one = kw_df > self.chk_size * 5
        first_push = True

        self.job_ids = []

        # chunk into appropriate sizes, to constrain number of ops in each job
        for df in daf_get.chunks(kw_df, self.chk_size):

            # Google requests that we wait 2 seconds between multiple pushes to the same account
            if big_one and not first_push:
                sleep(2)
            first_push = False

            job_id = self.mutations.modify_keywords(df.ops.apply(json.loads).tolist())
            self.job_ids.append(job_id)

            # Replace any negative signs with an 'N' so that HDF5 Store stops giving warnings.
            self.store.put('jobs/_' + job_id.replace('-', 'N'), df)
            self.logger.log(level=logging.DEBUG, ko="Pushed new job to google, id = {}".format(job_id))

        return self.job_ids