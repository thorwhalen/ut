__author__ = 'thorwhalen'

import numpy as np
from os import getcwd
from os.path import join
from numpy import mod
import json

def mk_arg_val_dict_from_sys_argv(sys_argv,
                                  convert_values_to_number_if_possible=False,
                                  convert_braced_strings_to_json=False):
    """
    Transform the list given by sys.argv (which parses the tokens of the command line) into a dict whose
    keys are the strings of the arguments prefixed with -, and the values are the arguments that follow
    (until the next "-" argument or the end of the argv list.

    Note that the value of a key argument will be a list if and only if there's more than one non-key argument follwing
    it.

    For example:
    >>> sys_argv = ['some_python_file.py', '-first', '1', \
                    '-second', 'foo', 'bar',\
                    '-js', '{"aa":3,"bb":null}']
    >>> mk_arg_val_dict_from_sys_argv(sys_argv, True, True) == {'first': 1, \
                                                                'second': ['foo', 'bar'], \
                                                                'js': {'aa': 3, 'bb': None}}
    True
    """
    sys_argv = sys_argv[1:]
    narg = len(sys_argv)
    arg_val_dict = {}

    if narg == 0:
        return arg_val_dict
    else:
        assert sys_argv[0].startswith('-'), "The list must start with a key arg (prefixed with '-')"
        current_key_arg = None
        for arg in sys_argv:
            if arg.startswith('-'):
                if current_key_arg is not None:
                    # add key and value(s) to the arg_val_dict
                    if len(current_value_list) == 0:
                        arg_val_dict[current_key_arg] = True
                    elif len(current_value_list) == 1:
                        arg_val_dict[current_key_arg] = current_value_list[0]
                    else:
                        arg_val_dict[current_key_arg] = current_value_list
                # and reinitialize key and value
                current_key_arg = arg[1:]
                current_value_list = []
            else:
                if convert_braced_strings_to_json and arg.startswith('{'):
                    arg = json.loads(arg)
                elif convert_values_to_number_if_possible:
                    arg = convert_to_number_if_possible(arg)
                current_value_list.append(arg)

        # add key and value(s) to the arg_val_dict
        if len(current_value_list) == 0:
            arg_val_dict[current_key_arg] = True
        elif len(current_value_list) == 1:
            arg_val_dict[current_key_arg] = current_value_list[0]
        else:
            arg_val_dict[current_key_arg] = current_value_list

        return arg_val_dict


def convert_to_number_if_possible(x):
    try:
        x = float(x)
        if x == int(x):
            x = int(x)
        return x
    except ValueError:
        return x



def full_filepath(rel_path):
    return join(getcwd(), rel_path)


def print_info(x, max_depth=30, print_contents=False, depth=0, tab=''):
    if depth <= max_depth:
        class_info = x.__class__
        if hasattr(x,'__name__'):
            print "%s%s %s" % (tab+'  ', x.__name__, type.mro(class_info)[0])
        else:
            print "%s%s" % (tab+'  ',type.mro(class_info)[0])
        new_depth = depth + 1
        if hasattr(x, '__dict__'):
            dict_info = x.__dict__
            if dict_info:
                tab = tab + '    '
                for k,v in dict_info.items():
                    print tab + '.' + k + ":"
                    # print "%s%s: %s" (tab, k, v.__class__)
                    print_info(v, max_depth=max_depth, print_contents=print_contents, depth=new_depth, tab=tab)
        if hasattr(x, '__self__'):
            print_info(x.__self__, max_depth=max_depth, print_contents=print_contents, depth=new_depth, tab=tab)
        if print_contents:
            contents_to_print = []
            if isinstance(x, dict):
                contents_to_print = x.keys()
            elif isinstance(x, list):
                contents_to_print = x
            if contents_to_print:
                contents_to_print = contents_to_print[:min(5,len(contents_to_print))]
                print tab + str(contents_to_print)


def is_an_iter(x):
    """
    this function identifies iterables that are not strings
    """
    return hasattr(x, '__iter__')

def is_callable(x):
    """
    this function identifies variables that are callable
    """
    return hasattr(x,'__call__')

def my_to_list(x):
    """
    to_list(x) blah blah returns [x] if x is not already a list, and x itself if it's already a list
    Use: This is useful when a function expects a list, but you want to also input a single element without putting this
    this element in a list
    """
    print "util.var.my_to_list() DEPRECIATED!!!: use util.ulist.ascertain_list() instead!!!"
    if not isinstance(x,list):
        if is_an_iter(x):
            x = list(x)
        else:
            x = [x]
    return x

def typeof(x):
    if isinstance(x,list):
        if len(x) > 0:
            unik_types = list(np.lib.unique([typeof(xx) for xx in x]))
            if len(unik_types)==1:
                return "list of " + unik_types[0]
            elif len(unik_types) <= 3:
                return "list of " + ", ".join(unik_types)
            else:
                return "list of various types"
        else:
            return "empty list"
    else:
        return type(x).__name__


# if __name__=="__main__":
#     print my_to_list('asdf')

