import time


class WithFeedback:
    """Context manager that will serve as a timer, with custom feedback prints (or logging, or any callback)
    >>> with WithFeedback():
    ...     time.sleep(0.5)
    <BLANKLINE>
    <BLANKLINE>
    Took 0.5 seconds
    >>> with WithFeedback("doing something...", "... finished doing that thing"):
    ...     time.sleep(0.5)
    doing something...
    ... finished doing that thing
    Took 0.5 seconds
    >>> with WithFeedback(verbose=False) as feedback:
    ...     time.sleep(1)
    >>> # but you still have access to some stats
    >>> _ = feedback  # For example: feedback.elapsed=1.0025296140000002 (start=1.159414532, end=2.161944146)
    """

    def __init__(self, start_msg="", end_msg="", verbose=True, print_func=print):
        self.start_msg = start_msg
        self.end_msg = end_msg
        self.verbose = verbose
        self.print_func = print_func  # change print_func if you want to log, etc. instead

    def print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            if len(args) > 0 or len(kwargs) > 0:
                return self.print_func(*args, **kwargs)

    def __enter__(self):
        self.print_if_verbose(self.start_msg)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        self.print_if_verbose(self.end_msg + "\n" + f"Took {self.elapsed:0.1f} seconds")

    def __repr__(self):
        return f"elapsed={self.elapsed} (start={self.start}, end={self.end})"
