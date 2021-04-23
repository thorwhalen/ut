import random

flatten = lambda l: [item for sublist in l for item in sublist]


def _train_test_keys_split(grouped_keys, n_train, if_insufficient_data='only_train'):
    groups = list(grouped_keys)
    if n_train > len(groups):
        if if_insufficient_data == 'only_train':
            return set(groups), set([])
        else:
            raise ValueError(f"Don't know how to handle if_insufficient_data: {if_insufficient_data}")
    else:
        train_groups = random.sample(groups, n_train)
        test_groups = set(groups) - set(train_groups)
        return (set(flatten([grouped_keys[g] for g in train_groups])),
                set(flatten([grouped_keys[g] for g in test_groups])))


def train_test_keys_split(grouped_keys, train_prop=0.8):
    return _train_test_keys_split(grouped_keys, int(len(grouped_keys) * train_prop))


def train_test_keys_leave_one_out_split(grouped_keys):
    return _train_test_keys_split(grouped_keys, len(grouped_keys) - 1)


#
# def group_keys(self, group_keys, keys=None):
#     """Make a util data structure that organizes the keys into separate universes.
#
#     Args:
#         keys: The keys to split
#         *group_keys: The sequence of group_keys that define how to group the keys.
#             Fundementally, a group_key is a function that takes a key and spits out a hashable value to group by
#             But if the specified group key is a string or a tuple of strings,
#             the function that groups by the those namedtuple attributes will be
#
#     Returns: A nested dictionary of group_keys whose leaves are the lists of the subsets of the input keys that
#         match the group_keys path.
#
#     Example:
#     >>> dacc = PreppedDacc()
#     >>> t = dacc.group_keys(['pump_type', ('pump_serial_number', 'session'), len])
#     >>> # groups the keys by pump_type, then by ('pump_serial_number', 'session'), then by the length.
#     >>> def nested_depth(d, _depth_so_far=0):
#     ...     if not isinstance(d, dict):
#     ...         return _depth_so_far
#     ...     else:
#     ...         return nested_depth(next(iter(d.values())), _depth_so_far + 1)
#     >>> nested_depth(t)
#     3
#     """
#     if keys is None:
#         keys = list(self.filterd_prepped_data)
#
#     _key_funcs = list()
#     for group_key in group_keys:
#         if isinstance(group_key, str):
#             # _key_funcs.append(lambda k: sub_namedtuple(k, tuple([group_key])))
#             _key_funcs.append(partial(sub_namedtuple, index=(group_key,)))
#         elif not callable(group_key):
#             _key_funcs.append(partial(sub_namedtuple, index=tuple(group_key)))
#         else:
#             _key_funcs.append(group_key)
#
#     return regroupby(keys, *_key_funcs)


def random_train_test_split_keys(self, keys=None, test_size=0.2,
                                 group_key=('pump_type', 'test_phase'),
                                 cannot_be_separated='pump_serial_number',
                                 keep_phase_9_keys=False):
    """Split keys randomly in view of train/test testing.

    Args:
        keys: The keys to split (all of dacc.filterd_prepped_data by default)
        test_size: A proportion to assign to test
        group_key: The key function of fields used to group by.
            This is usually used so as to separate training sets
        cannot_be_separated: The field of fields that should not be separated in train and test:
            That is, keys agreeing on these fields should be either entirely in train or in test.

    Returns:
        A {group: (train_key_set, test_key_set), ...} dict

    """

    if keys is None:
        keys = list(self.filterd_prepped_data)

    elif callable(keys):
        keys_filter = keys
        keys = filter(keys_filter, self.filterd_prepped_data)

    if not keep_phase_9_keys:
        # Systematically remove phase 9 keys
        keys = list(keys)
        n_keys = len(keys)
        keys = list(filter(lambda x: x.test_phase != 9, keys))
        if len(keys) != n_keys:
            from warnings import warn
            warn(f"I removed {n_keys - len(keys)} test_phase==9 of the {n_keys} keys you specified")

    if group_key is not None:
        groups = self.group_keys([group_key, cannot_be_separated], keys)
        n_groups = len(groups)
    else:
        n_groups = len(keys)

    if isinstance(test_size, float):
        n_train = int(n_groups * (1 - test_size))
    elif isinstance(test_size, int):
        n_train = n_groups - test_size
    else:
        raise TypeError(f"I don't recognize that type of test_size: {test_size}")

    def _train_test_keys_split_output_as_dict(*args, **kwargs):
        train, test = _train_test_keys_split(*args, **kwargs)
        return {'train': train, 'test': test}

    if group_key is not None:
        return {group_key: _train_test_keys_split_output_as_dict(grouped_keys, n_train)
                for group_key, grouped_keys in groups.items()}
    else:
        return _train_test_keys_split_output_as_dict(keys, n_train)
