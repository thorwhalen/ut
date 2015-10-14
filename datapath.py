__author__ = 'thorwhalen'


#from os import environ # does this load the whole array? Can we just take MS_DATA instead?
from ut.util.importing import get_environment_variable

#MS_DATA = os.environ['MS_DATA']


def datapath(filename=''):
    # NOTE: Tried os.path.join first, but noticed that if filename started with a /, we didn't get the MS_DATA root.
    # so for simplicity and (??) performance, chose to use concatination

    # protecting filename from None
    filename = filename or ''
    return get_environment_variable('MS_DATA') + filename


# import os
# from sys import argv
#
# def main(filename=''):
#     return os.path.join(os.environ('MS_DATA'),filename)
#
# if __name__ == "__main__":
#     main(argv[0])


