import time


class TimerAndFeedback:
    """Context manager that will serve as a timer, with custom feedback prints (or logging, or any callback)
    >>> with TimerAndFeedback():
    ...     time.sleep(0.5)
    Took 0.5 seconds
    >>> with TimerAndFeedback("doing something...", "... finished doing that thing"):
    ...     time.sleep(0.5)
    doing something...
    ... finished doing that thing
    Took 0.5 seconds
    >>> with TimerAndFeedback(verbose=False) as feedback:
    ...     time.sleep(1)
    >>> # but you still have access to some stats through feedback object (like elapsed, started, etc.)
    """

    def __init__(self, start_msg="", end_msg="", verbose=True, print_func=print):
        self.start_msg = start_msg
        if end_msg:
            end_msg += '\n'
        self.end_msg = end_msg
        self.verbose = verbose
        self.print_func = print_func  # change print_func if you want to log, etc. instead

    def print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            if len(args) > 0 and len(args[0]) > 0:
                return self.print_func(*args, **kwargs)

    def __enter__(self):
        self.print_if_verbose(self.start_msg)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        self.print_if_verbose(self.end_msg + f"Took {self.elapsed:0.1f} seconds")

    def __repr__(self):
        return f"elapsed={self.elapsed} (start={self.start}, end={self.end})"
