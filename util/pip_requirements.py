

from ut.pfile.to import string as file_to_string
import pandas as pd
from numpy import array
import os
from warnings import warn


def requirement_file_to_df(filepath):
    s = file_to_string(filepath)
    t = [xx for xx in [x.split('==') for x in s.split('\n')] if len(xx) == 2]
    return pd.DataFrame([{'pkg': x[0], 'version': x[1]} for x in t])


def requirements_comparison_df_only_when_different(requirements_filepath_1, requirements_filepath_2):
    c = requirements_comparison_df(requirements_filepath_1, requirements_filepath_2)
    c = c[c['version_x'] != c['version_y']]
    return c


def requirements_comparison_df(requirements_filepath_1, requirements_filepath_2):
    r1 = requirement_file_to_df(requirements_filepath_1)
    r2 = requirement_file_to_df(requirements_filepath_2)
    c = r1.merge(r2, how='outer', on='pkg')
    return c


str_to_num_key = array([1e12, 1e6, 1e3, 1]).astype(int)


def version_str_to_num(version_str):
    try:
        if isinstance(version_str, str):
            num = 0
            for i, v in enumerate(version_str.split('.')):
                num += int(v) * str_to_num_key[i]
            return num
        else:
            return None
    except:
        return None


def requirements_comparison_objects(requirements_filepath_1, requirements_filepath_2):
    c = requirements_comparison_df_only_when_different(requirements_filepath_1, requirements_filepath_2)

    c, missing_1, missing_2 = rm_empty_version_entries(c)

    v1_num = list()
    v2_num = list()
    for _, row in c.iterrows():
        v1 = version_str_to_num(row['version_x'])
        v2 = version_str_to_num(row['version_y'])
        if v1 and v2:
            v1_num.append(v1)
            v2_num.append(v2)
        else:
            print(("!!! Couldn't get the version NUMBER for {}\n".format(dict(row))))
    v1_num = array(v1_num)
    v2_num = array(v2_num)

    v1_greater_than_v2_df = c.ix[v1_num > v2_num]
    v2_greater_than_v1_df = c.ix[v2_num > v1_num]

    return missing_1, missing_2, v1_greater_than_v2_df, v2_greater_than_v1_df


def rm_empty_version_entries(comp_df):
    lidx = comp_df['version_x'].isnull()
    missing_1 = list(comp_df.ix[lidx, 'pkg'])
    comp_df = comp_df.ix[~lidx]

    lidx = comp_df['version_y'].isnull()
    missing_2 = list(comp_df.ix[lidx, 'pkg'])
    comp_df = comp_df.ix[~lidx]

    return comp_df, missing_1, missing_2


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

    print(("\n-------- Missing in {}:".format(name1)))
    print(missing_1)

    print(("\n-------- Missing in {}:".format(name2)))
    print(missing_2)

    print(("\n-------- {} in advance of {}:".format(name1, name2)))
    print((v1_greater_than_v2_df.rename(columns={'version_x': name1, 'version_y': name2})))

    print(("\n-------- {} in advance of {}:".format(name2, name1)))
    print((v2_greater_than_v1_df.rename(columns={'version_x': name1, 'version_y': name2})))


def get_requirements_to_update_second_requirements_when_behind_first(
        requirements_filepath_1, requirements_filepath_2):
    missing_1, missing_2, v1_greater_than_v2_df, v2_greater_than_v1_df = \
        requirements_comparison_objects(requirements_filepath_1, requirements_filepath_2)
    s = ''
    for i, row in v1_greater_than_v2_df.iterrows():
        s += row['pkg'] + '==' + row['version_x'] + "\n"

    return s


def updated_requirements_2_with_requirements_1_that_are_ahead(
        requirements_filepath_1, requirements_filepath_2):

    r1 = requirement_file_to_df(requirements_filepath_1).set_index('pkg')['version']
    r2 = requirement_file_to_df(requirements_filepath_2).set_index('pkg')['version']

    s = ''
    for pkg, v2_str in r2.items():
        v2 = version_str_to_num(v2_str)
        v1_str = r1.get(pkg)
        v1 = version_str_to_num(v1_str)
        if v1 and v1 > v2:
            s += pkg + '==' + v1_str + "\n"
        else:
            s += pkg + '==' + v2_str + "\n"
    return s
    #
    #
    # comp_df = requirements_comparison_df(requirements_filepath_1, requirements_filepath_2)
    #
    # lidx = comp_df['version_y'].isnull()
    # comp_df = comp_df.ix[~lidx]
    #
    # s = ''
    # for _, row in comp_df.iterrows():
    #     v1 = version_str_to_num(row['version_x'])
    #     v2 = version_str_to_num(row['version_y'])
    #
    #     if v2:
    #         if v1 > v2:
    #             s += row['pkg'] + '==' + row['version_x'] + "\n"
    #         else:
    #             s += row['pkg'] + '==' + row['version_y'] + "\n"
    # return s
