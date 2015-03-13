__author__ = 'thorwhalen'

import requests
from serialize.khan_logger import KhanLogger
import logging


class SimpleRequest(object):

    def __init__(self, log_file_name=None, log_level=logging.INFO):
        full_log_path_and_name = KhanLogger.default_log_path_with_unique_name(log_file_name)
        self.logger = KhanLogger(file_path_and_name=full_log_path_and_name, level=log_level)

    def slurp(self, url):

        r = requests.get(url, timeout=30.0)

        if not r.ok:
            self.logger.log(level=logging.WARN, simple_request="HTTP Error: {} for url {}".format(r.status_code, url))
        else:
            self.logger.log(level=logging.INFO, simple_request="Slurped url {}".format(url))
            return r.text

if __name__ == '__main__':
    sr = SimpleRequest()