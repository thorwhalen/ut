__author__ = 'mattjmorris'

import logging
import traceback
import json
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import os
import numpy as np
from serialize.amazon_sender import AmazonSender
from collections import OrderedDict


class KhanLogger(object):

    @classmethod
    def get_most_recent_log_as_df(cls):
        return cls.get_log_as_df(cls.most_recent_log())

    @classmethod
    def get_most_recent_error(cls):
        return cls.get_last_error(cls.most_recent_log())

    @classmethod
    def get_log_as_df(cls, file_name_and_path):

        with open(file_name_and_path) as f:
            content = f.readlines()

        # only pull in rows that comply with json
        data = [json.loads(d) for d in content if d.startswith('{')]

        if data:
            df = DataFrame(data)
            df['dt'] = pd.to_datetime(df['dt'])
            df = df.set_index('dt')
            # Get columns we are interested in to show up in order we want them in
            cols_of_interest = [col for col in ['msg', 'origin', 'level', 'error_traceback'] if col in df.columns]
            cols_not_of_interest = list(set(df.columns) - set(cols_of_interest))
            df = df[cols_of_interest + cols_not_of_interest]
        else:
            df = DataFrame()

        return df.fillna('')

    @classmethod
    def get_last_error(cls, file_name_and_path):
        df = cls.get_log_as_df(file_name_and_path)
        if df:
            df = df[df['error_traceback'].notnull()]
            df.sort_index(ascending=False, inplace=True)
            return df.iloc[0]['error_traceback']
        else:
            return ''

    @classmethod
    def most_recent_log(cls, base_folder=None):
        logdir = base_folder or os.getenv('KHAN_LOG_FOLDER')
        logfiles = sorted([f for f in os.listdir(logdir) if f.endswith('.log')])
        most_recent_log = os.path.join(logdir, logfiles[-1])
        return most_recent_log

    @classmethod
    def move_logs_to_backup(cls, base_folder=None):

        logdir = base_folder or os.getenv('KHAN_LOG_FOLDER')
        backup_folder = os.path.join(logdir, 'bak')
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)
        # delete out old ones so only most recent old one is ever there
        else:
            for f in os.listdir(backup_folder):
                file_path = os.path.join(backup_folder, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

        for the_file in os.listdir(logdir):
            try:
                if the_file.endswith(".log"):
                    os.rename(os.path.join(logdir, the_file), os.path.join(backup_folder, the_file))
            except Exception as e:
                print(e)

    def __init__(self, level=logging.INFO, file_path=None, file_name='main.log', make_file_name_unique=False, mode='a', origin=''):
        """
        Main purpose of this logger is to make sure everything gets logged in JSON format, so that it can easily
        be read back in the form of a Pandas dataframe.

        Deprecated parameter: namespace. Remove as soon as able to verify it is no longer in use.
        Mode: w for write, a for append. Defaults to w.
        """
        #
        #
        #if os.path.exists(file_path):
        #    name_and_path = file_path
        #elif file_name:
        #    file_name
        #    logs = [f for f in os.listdir(os.environ['KHAN_LOG_FOLDER']) if f.endswith(".log")]

        # if a file already exists, use it:
        logs = [f for f in os.listdir(os.getenv('KHAN_LOG_FOLDER')) if f.endswith(".log")]
        if logs:
            name_and_path = os.path.join(os.getenv('KHAN_LOG_FOLDER'), logs[0])
        else:
            name_and_path = self._make_file_name_and_path(file_name, file_path, make_file_name_unique)



        self.filename_and_path = name_and_path

        generic_folder = os.path.join(os.getenv('KHAN_LOG_FOLDER'), 'generic')
        if not os.path.exists(generic_folder):
            os.makedirs(generic_folder)
        generic_log_name_and_path = os.path.join(generic_folder, "generic.log")
        logging.basicConfig(filemode='w', filename=generic_log_name_and_path, level=level, format='%(message)s')

        # This next line uses the name_and_path as a unique namespace
        self.logger = logging.getLogger(name_and_path)
        #self.logger.setLevel(level)
        #self.logger
        self.logger.handlers = []
        fh = logging.FileHandler(name_and_path, mode='a')
        fh.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        #self.logger

        self.origin = origin

        self.email = AmazonSender()
        self.num_errors = 0

    def log(self, level=logging.DEBUG, msg=None, origin=None, error_obj=None, **kwargs):

        origin = origin or self.origin

        # construct a dict to log
        log_dict = OrderedDict()

        if msg:
            log_dict['msg'] = msg
        log_dict['origin'] = origin
        log_dict['level'] = logging.getLevelName(level)
        log_dict['dt'] = datetime.now().isoformat()
        log_dict.update(kwargs)

        # MJM: the traceback object holds on to errors, so only ask it for info if we are logging an error
        if level == logging.ERROR:
            # If there was an exception, let's grab it here
            tb = traceback.format_exc()
            if tb and tb!="None\n":
                log_dict['error_traceback'] = tb
        elif error_obj:
            log_dict['error_traceback'] = str(error_obj)

        self.logger.log(level=level, msg=json.dumps(log_dict))

    def debug(self, msg, **kwargs):
        self.log(level=logging.DEBUG, msg=msg, **kwargs)

    def info(self, msg, **kwargs):
        self.log(level=logging.INFO, msg=msg, **kwargs)

    def warn(self, msg, error_obj=None, **kwargs):
        self.log(level=logging.WARN, msg=msg, error_obj=error_obj, **kwargs)

    def error(self, msg, error_obj=None, send_email=True, **kwargs):
        """
        For message, it is recommended you pass in the message from the error_obj object (if there is one), or else a
        custom message that includes the error_obj message.
        The traceback will be automatically extracted from the traceback object.
        If sending email, it is recommended you pass in the original error_obj object
        """
        self.log(level=logging.ERROR, msg=msg, error_obj=error_obj, **kwargs)
        if send_email:
            body_html = "<p> Last log entries: </p>"
            body_html += "</br>"

            full_error = None

            try:
                df = self.get_current_log_as_df()
                body_html += df[-10:].to_html()
                self.num_errors += 1

                try:
                    df_sorted = df.sort_index(ascending=False)
                    full_error = df_sorted.iloc[0]['error_traceback']
                except:
                    pass
            except Exception as e:
                self.warn("Could not generate error df", error_obj=e)

            error_html = full_error or str(error_obj) or msg
            # force spacing
            error_html = error_html.replace(' ', '&nbsp;')
            # put brs where there had been newlines
            error_html = error_html.replace('\n', '<br />')
            error_html = "<p>" + error_html + "</p>"
            error_html += body_html

            if self.num_errors == 10:
                self.email.send_email(subject='10th KHAN Error, last one sending', text="Error", html=error_html)
            elif self.num_errors < 10:
                self.email.send_email(subject='KHAN Error', text="Error", html=error_html)

    # DEPRECATED
    #def structured_log(self, level=logging.DEBUG, origin=None, msg='', **kwargs):
    #    origin = origin or self.origin or ''
    #    self.log(level=level, origin=origin, msg=msg, **kwargs)

    # DEPRECATED
    #def id_and_reason_log(self, id, reason, level=logging.ERROR):
    #    self.log(level=level, id=id, reason=reason)

    def get_current_log_as_df(self):
        """
        Calls the class-level method for getting df of of a file, passing in path of current file
        """
        return self.get_log_as_df(self.filename_and_path)

    def get_current_error(self):
        """
        Calls the class-level method for getting last error from a file, passing in path of current file
        """
        return self.get_last_error(self.filename_and_path)

    def _make_file_name_and_path(self, file_name, file_path, make_file_name_unique):

        file_path = file_path or os.getenv('KHAN_LOG_FOLDER')

        if make_file_name_unique:
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            name = os.path.splitext(file_name)[0]
            extension = os.path.splitext(file_name)[1]
            if name:
                name += "_"
            file_name = name + time_str + extension

        if not os.path.splitext(file_name)[1]:
            file_name += '.log'

        full_name_and_path = os.path.join(file_path, file_name)

        return full_name_and_path

    # DEPRECATED
    def llog(self, level=None, **kwargs):
        """
        exactly like log, except it defaults to self.logger.level instead of to logging.DEBUG
        (llog for Level-LOG)
        """

        # default to object's logger.level
        level = level or self.logger.level

        # call log
        self.log(level=level, **kwargs)

    # DEPRECATED
    @classmethod
    def get_reason_stats(cls, log_df):
        nRows = len(log_df)
        stats_df = log_df[['reason']].dropna()
        nNanRows = nRows - len(stats_df)
        stats_df = stats_df.groupby(['reason']).count()
        stats_df = stats_df.rename(columns={'reason': 'count'})
        #stats_df = ch_col_names(stats_df, ['count'], ['reason'])
        #return stats_df
        stats_df = pd.concat([stats_df,
                              pd.DataFrame({'reason': ['NAN'], 'count': [nNanRows]}).set_index('reason')])
        #return stats_df.order(ascending=False)
        return stats_df.sort(columns=['count'], ascending=False)

    # DEPRECATED
    @classmethod
    def get_no_error_rows(cls, log_df):
        return log_df[~np.isnan(log_df.reason)]