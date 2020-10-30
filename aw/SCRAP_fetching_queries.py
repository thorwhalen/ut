__author__ = 'thorwhalen'

import datetime
from . import reporting as rp
import pandas

#settings
save_folder = '/D/Dropbox/dev/py/data/query_data/'
account_list = rp.get_account_id('dict')
account_list  = list(account_list.keys())
numOfDays = 60
report_query_str = rp.mk_report_query_str(
    varList='q_iipic',
    start_date= numOfDays)
def save_file(account):
    return "{}{}-{}-{}days.p".format(
        save_folder,
        account,
        datetime.date.today().strftime("%Y%m%d"),
        numOfDays)


def getting_a_bunch_of_queries():
        # running that shit
    i = 0
    account_list = ['test']
    for account in account_list:
        i = i + 1
        print("({}/{}) downloading {}".format(i,len(account_list),account))
        report_downloader = rp.get_report_downloader(account)
        df = rp.download_report(
            report_downloader=report_downloader,
            report_query_str=report_query_str,
            download_format='df')
    saveFile = save_file(account)
    print("   Saving to {}".format(saveFile))
    df.save(saveFile)


# run this shit
getting_a_bunch_of_queries()