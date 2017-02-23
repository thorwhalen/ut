from __future__ import division

from ut.pfile.to import string as file_to_string
import pandas as pd
from numpy import array
import os


def requirement_file_to_df(filepath):
    s = file_to_string(filepath)
    t = filter(lambda xx: len(xx) == 2, map(lambda x: x.split('=='), s.split('\n')))
    return pd.DataFrame(map(lambda x: {'pkg': x[0], 'version': x[1]}, t))


def requirements_comparison_df(requirements_filepath_1, requirements_filepath_2):
    r1 = requirement_file_to_df(requirements_filepath_1)
    r2 = requirement_file_to_df(requirements_filepath_2)
    c = r1.merge(r2, how='outer', on='pkg')
    c = c[c['version_x'] != c['version_y']]
    return c


str_to_num_key = array([1e12, 1e6, 1e3, 1]).astype(int)


def version_str_to_num(version_str):
    if isinstance(version_str, basestring):
        num = 0
        for i, v in enumerate(version_str.split('.')):
            num += int(v) * str_to_num_key[i]
        return num
    else:
        return None


def requirements_comparison_objects(requirements_filepath_1, requirements_filepath_2):
    c = requirements_comparison_df(requirements_filepath_1, requirements_filepath_2)

    lidx = c['version_x'].isnull()
    missing_1 = list(c.ix[lidx, 'pkg'])
    c = c.ix[~lidx]

    lidx = c['version_y'].isnull()
    missing_2 = list(c.ix[lidx, 'pkg'])
    c = c.ix[~lidx]

    v1_num = array(map(version_str_to_num, c['version_x']))
    v2_num = array(map(version_str_to_num, c['version_y']))

    v1_greater_than_v2_df = c.ix[v1_num > v2_num]
    v2_greater_than_v1_df = c.ix[v2_num > v1_num]

    return missing_1, missing_2, v1_greater_than_v2_df, v2_greater_than_v1_df


def file_unique_identifiers(f1, f2):
    root1 = os.path.splitext(f1)[0]
    root2 = os.path.splitext(f2)[0]

    cand1 = ''
    cand2 = ''
    while len(root1) > 0 or len(root2) > 0:
        root1, next1 = os.path.split(root1)
        root2, next2 = os.path.split(root2)
        if len(cand1) > 0:
            cand1 = next1 + '/' + cand1
            cand2 = next2 + '/' + cand2
        else:
            cand1 = next1
            cand2 = next2
        if cand1 != cand2:
            break

    return cand1, cand2


def print_requirements_comparison(requirements_filepath_1, requirements_filepath_2):
    missing_1, missing_2, v1_greater_than_v2_df, v2_greater_than_v1_df = \
        requirements_comparison_objects(requirements_filepath_1, requirements_filepath_2)

    name1, name2 = file_unique_identifiers(requirements_filepath_1, requirements_filepath_2)

    print("\n-------- Missing in {}:".format(name1))
    print(missing_1)

    print("\n-------- Missing in {}:".format(name2))
    print(missing_2)

    print("\n-------- {} in advance of {}:".format(name1, name2))
    print(v1_greater_than_v2_df.rename(columns={'version_x': name1, 'version_y': name2}))

    print("\n-------- {} in advance of {}:".format(name2, name1))
    print(v2_greater_than_v1_df.rename(columns={'version_x': name1, 'version_y': name2}))


def get_requirements_to_update_second_requirements_when_behind_first(
        requirements_filepath_1, requirements_filepath_2):
    missing_1, missing_2, v1_greater_than_v2_df, v2_greater_than_v1_df = \
        requirements_comparison_objects(requirements_filepath_1, requirements_filepath_2)
    s = ''
    for i, row in v1_greater_than_v2_df.iterrows():
        s += row['pkg'] + '==' + row['version_x'] + "\n"

    return s