# import os
# import sys
# log_filename = os.path.basename(sys.argv[0]).replace('.py', '.log')
# log_filepath = os.path.join(os.environ['DEFAULT_LOG_FOLDER'], log_filename)
# # log_filepath = os.path.join(os.environ['DEFAULT_LOG_FOLDER'], __name__.split('.')[-1])
# print ""
# print "-----------------------------------------------"
# print "LOG FILEPATH: %s" % log_filepath
# import logging
# logging.basicConfig(filename=log_filepath, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

# # from ut.util.imports.ipython_utils import *
# from ut.dacc.mong.com import MongoStruct
# # import pandas as pd
from datetime import datetime

# from random import randint
import time

# import pymongo
from ut.webscrape.misc.amazon import AmazonBookWatch
import ut as ms
import ut.dacc.mong.util
from ut.serialize.amazon_sender import AmazonSender
import pandas as pd

frequency_in_hours = 1
time_of_day_to_send_email = 8

spec = {
    'Vanessa Able': {
        'title_country_list': [
            {'title': 'Never mind the bullocks', 'country': 'co.uk'},
            {'title': 'Never mind the bullocks', 'country': 'com'},
            {'title': 'The Nanologues', 'country': 'in'},
        ],
        'subscriber_emails': ['vanessa.able@gmail.com', 'thor@mscoms.com'],
    }
}

# 'Miriam Williams': {'title_country_list': [
#     {'title': "Heaven's Harlots (Paperback)", 'country': 'com'},
#     {'title': "Heaven's Harlots (Hardcover)", 'country': 'com'},
#     {'title': "Women on Ice", 'country': 'com'}
# ],
#             'subscriber_emails': ['miriamboeri@gmail.com', 'thor@mscoms.com']
# }

# 'Julia Cooke': {'title_country_list': [
#     {'title': 'The Other Side of Paradise', 'country': 'com'},
#     {'title': 'The Other Side of Paradise', 'country': 'co.uk'}
# ],
#             'subscriber_emails': ['julia.cooke@gmail.com', 'thor@mscoms.com']
# }

# subscriber_emails = ['vanessa.able@gmail.com', 'thor@mscoms.com']

abw = AmazonBookWatch()
while True:
    info_list = abw.get_book_statuses()
    abw.db.insert(info_list)

    if datetime.now().hour == time_of_day_to_send_email:
        try:
            for author, ispec in spec.items():
                html = abw.mk_html_report(
                    title_country_list=ispec['title_country_list']
                )
                amazon_sender = AmazonSender(to_addresses=ispec['subscriber_emails'])
                amazon_sender.send_email(
                    subject="{author}'s Book Watch".format(author=author), html=html
                )
        except Exception as e:
            error_amazon_sender = AmazonSender(to_addresses='thor@mscoms.com')
            error_amazon_sender.send_email(subject='Woops', html=e.message)

    time.sleep(frequency_in_hours * 60 * 60)
