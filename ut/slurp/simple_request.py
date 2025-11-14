__author__ = 'thorwhalen'

import requests
from serialize.khan_logger import KhanLogger
import logging


class SimpleRequest:
    def __init__(self, log_file_name=None, log_level=logging.INFO):
        full_log_path_and_name = KhanLogger.default_log_path_with_unique_name(
            log_file_name
        )
        self.logger = KhanLogger(
            file_path_and_name=full_log_path_and_name, level=log_level
        )

    def slurp(self, url):

        r = requests.get(url, timeout=30.0)

        if not r.ok:
            self.logger.log(
                level=logging.WARN,
                simple_request=f'HTTP Error: {r.status_code} for url {url}',
            )
        else:
            self.logger.log(
                level=logging.INFO, simple_request=f'Slurped url {url}'
            )
            return r.text


if __name__ == '__main__':
    sr = SimpleRequest()
