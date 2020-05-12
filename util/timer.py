__author__ = 'thorwhalen'

import time


class Timer(object):
    def __init__(self):
        self.start_time = time.time()
        self.last_tick_time = time.time()

    def reset_timer(self):
        self.start_time = time.time()
        self.last_tick_time = self.start_time

    def elapsed_time(self):
        return time.time() - self.start_time

    def tick_time(self):
        tick = time.time() - self.last_tick_time
        self.last_tick_time = time.time()
        return tick
