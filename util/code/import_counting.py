from numpy import unique
from collections import Counter
import re
import inspect
import os
import subprocess
from io import StringIO
import pandas as pd

from ut.util.code.packages import get_module_name, read_requirements

from ut.pfile.iter import get_filepath_iterator

module_import_regex_tmpl = "(?<=from) {package_name}|(?<=[^\s]import) {package_name}"

any_module_import_regex = re.compile(module_import_regex_tmpl.format(package_name='\w+'))


def mk_single_package_import_regex(module_name):
    return re.compile(module_import_regex_tmpl.format(package_name=module_name))


def mk_multiple_package_import_regex(module_names):
    if isinstance(module_names, str):
        module_names = [module_names]
    return re.compile('|'.join([mk_single_package_import_regex(x).pattern for x in module_names]))


def imports_in_module(module):
    """
    Get a list of strings showing what is imported in a module.
    :param module: An actual module object the file of the module (as given by inspect.getfile(module)
    :return: A list of strings showing the imported objects (modules, functions, variables, classes...)

    >>> print('\\n'.join(imports_in_module(__file__)))
    StringIO.StringIO
    collections.Counter
    inspect
    numpy.unique
    os
    pandas
    re
    subprocess
    ut.pfile.iter.get_filepath_iterator
    ut.util.code.packages.get_module_name
    ut.util.code.packages.read_requirements
    """
    if not isinstance(module, str):
        module = inspect.getfile(module)
        if module.endswith('c'):
            module = module[:-1]  # remove the 'c' of '.pyc'
    t = subprocess.check_output(['sfood-imports', '-u', module])
    return [x for x in t.split('\n') if len(x) > 0]


def base_modules_used_in_module(module):
    """
    Get a list of strings showing what base modules that are imported in a module.
    :param module: An actual module object the file of the module (as given by inspect.getfile(module)
    :return: A list of strings showing the imported base modules (i.e. the X of import X.Y.Z or from X.Y import Z).
    >>> base_modules_used_in_module(__file__)
    ['StringIO', 'collections', 'inspect', 'numpy', 'os', 'pandas', 're', 'subprocess', 'ut']
    """
    return list(unique([re.compile('\w+').findall(x)[0] for x in imports_in_module(module)]))


def base_module_imports_in_module_recursive(module):
    """
    Get a list of strings showing what base modules that are imported in a module, recursively.
    It's the recursive version of the base_modules_used_in_module function.
    Recursive in the sense that if module is a package module (i.e. containing a __init__.py and further submodules),
    the base_modules_used_in_module function will be applied to all .py files under the mother folder.
    Function returns a count (Counter object) of the number of modules where each base module was found.
    :param module: An actual module object the file of the module (as given by inspect.getfile(module)
    :param module_names: Modules to filter for.
        None: Will grab all modules
        A list or tuple: Of modules to grab
        If not will assume module_names is a regex to apply to find module names
    :return:
    """
    # if module_names is None:
    #     module_names = any_module_import_regex
    # elif isinstance(module_names, (tuple, list)):
    #     module_names = mk_multiple_package_import_regex(module_names)

    if inspect.ismodule(module):
        module = inspect.getsourcefile(module)
    if module.endswith('__init__.py'):
        module = os.path.dirname(module)

    if os.path.isdir(module):
        c = Counter()
        it = get_filepath_iterator(module, pattern='.py$')
        next(it)  # to skip the seed module itself, and not get into an infinite loop
        for _module in it:
            try:
                c.update(base_module_imports_in_module_recursive(_module))
            except Exception as e:
                if 'sfood-imports' in e.args[1]:
                    raise RuntimeError("You don't have sfood-imports installed (snakefood), so I can't do my job")
                else:
                    print(("Error with module {}: {}".format(_module, e)))
        return c
    elif not os.path.isfile(module):
        raise ValueError("module file not found: {}".format(module))

    return Counter(base_modules_used_in_module(module))
    # with open(module) as fp:
    #     module_contents = fp.read()
    # return Counter(map(lambda x: x[1:], unique(module_names.findall(module_contents))))


def requirements_packages_in_module(module, requirements=None):
    if requirements is None:
        requirements = list(pip_licenses_df(include_module_name=False)['package_name'])
    elif isinstance(requirements, str) and os.path.isfile(requirements):
        with open(requirements) as fp:
            requirements = fp.read().splitlines()

    p = re.compile('^[^=]+')
    module_names = list()
    for x in requirements:
        try:
            xx = p.findall(x)
            if xx:
                module_name = get_module_name(xx[0])
                module_names.append(module_name)
        except Exception as e:
            print(("Error with {}\n  {}".format(x, e)))

    return base_module_imports_in_module_recursive(module, module_names=requirements)


word_or_letter_p = re.compile('\w')
at_least_two_spaces_p = re.compile('\s{2,}')


def pip_licenses_df(package_names=None, include_module_name=True, on_module_search_error=None):
    """
    Get a dataframe of pip packages and licences
    :return:
    """
    pip_licenses_output = subprocess.check_output(['pip-licenses'])

    t = list(map(str.strip,
                 list(filter(word_or_letter_p.search,
                             pip_licenses_output.split('\n')))))
    t = [at_least_two_spaces_p.sub('\t', x) for x in t]
    t = '\n'.join(t)

    df = pd.read_csv(StringIO(t), sep='\t')
    df = df.rename(columns={'Name': 'package_name', 'Version': 'version', 'License': 'license'})
    if include_module_name:
        df['module'] = [get_module_name(x, on_error=on_module_search_error) for x in df['package_name']]
        df = df[['module', 'package_name', 'version', 'license']]  # reorder
    if package_names is not None:
        df = df[df['package_name'].isin(package_names)]
    return df
