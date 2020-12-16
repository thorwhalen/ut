import os
import sys
import inspect
import importlib
import importlib.util
from dataclasses import dataclass
import ast
import site
from enum import Enum
from collections import namedtuple
import inspect


def is_from_module(obj, module):
    """Check if an object "belongs" to a module.

    >>> import collections
    >>> is_from_module(collections.ChainMap, collections)
    True
    >>> is_from_module(is_from_module, collections)
    False
    """
    return getattr(obj, '__module__', '').startswith(module.__name__)


def second_party_names(module, obj_filt=None):
    """Generator of module attribute names that point to object the module actually defines.

    :param module: Module (object)
    :param obj_filt: Boolean function applied to object to filter it in
    :return:

    >>> from ut.util.code import modules
    >>> sorted(second_party_names(modules))[:5]
    ['DOTPATH', 'FILEPATH', 'FOLDERPATH', 'LOADED', 'ModuleSpecKind']
    >>> sorted(second_party_names(modules, callable))[:5]
    ['ModuleSpecKind', 'coerce_module_spec', 'get_imported_module_paths', 'is_from_module', 'is_module_dotpath']
    >>> sorted(second_party_names(modules, lambda obj: isinstance(obj, type)))
    ['ModuleSpecKind']
    """
    obj_filt = obj_filt or (lambda x: x)
    for attr in filter(lambda a: not a.startswith('_'), dir(module)):
        obj = getattr(module, attr)
        if is_from_module(obj, module) and obj_filt(obj):
            yield attr


class ModuleSpecKind(Enum):
    LOADED = 1  # a loaded import object
    DOTPATH = 2  # a dot-separated string path to the module (e.g. sklearn.decomposition.pca
    PATH = 3  # a list-like of the names of the path to the module (e.g. ('sklearn', 'decomposition', 'pca')
    FILEPATH = 4  # path to the .py of the module
    FOLDERPATH = 5  # path to the folder containing the __init__.py of the module


LOADED, DOTPATH, PATH, FILEPATH, FOLDERPATH = \
    map(lambda a: getattr(ModuleSpecKind, a),
        ['LOADED', 'DOTPATH', 'PATH', 'FILEPATH', 'FOLDERPATH'])


def is_module_dotpath(dotpath):
    """Checks if a dotpath points to a module. """
    try:
        spec = importlib.util.find_spec(dotpath)
        if spec is not None:
            return True
    except ModuleNotFoundError:
        pass
    return False


def module_spec_kind(module_spec):
    if inspect.ismodule(module_spec):
        return LOADED
    elif isinstance(module_spec, str):
        if is_module_dotpath(module_spec):
            return DOTPATH
        elif os.path.isfile(module_spec):
            return FILEPATH
        elif os.path.isdir(module_spec) \
                and os.path.isfile(os.path.join(module_spec, '__init__.py')):
            return FOLDERPATH
    # if you got so far and no match was found...
    raise TypeError(f"Couldn't figure out the module specification kind: {module_spec}")


loaded_module_from_dotpath = importlib.import_module


def loaded_module_from_dotpath_and_filepath(dotpath, filepath):
    """Get module object from file path and module dotpath"""
    module_spec = importlib.util.spec_from_file_location(dotpath, filepath)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def module_path(module):
    try:
        _module_path = module.__path__[0]
    except:
        _module_path = inspect.getfile(module)
        if _module_path.endswith('__init__.py'):
            _module_path = module_path[:len('__init__.py')]
    return _module_path


def submodules(module, on_error='print', prefix=None):
    from py2store.filesys import FileCollection

    prefix = prefix or site.getsitepackages()[0]

    _module_path = module_path(module)
    assert os.path.isdir(_module_path), f"Should be a directory: {module_path}"

    for filepath in FileCollection(_module_path, '{f}.py', max_levels=0):
        dotpath = '.'.join(filepath[len(prefix):(-len('.py'))].split(os.path.sep))
        if dotpath.startswith('.'):
            dotpath = dotpath[1:]
        if not dotpath.endswith('__init__'):
            try:
                yield loaded_module_from_dotpath_and_filepath(dotpath, filepath)
            except Exception as e:
                if on_error == 'print':
                    print(f"{e} ({dotpath}: {filepath})")
                elif on_error == 'raise':
                    raise


def objects_of_module(module, max_levels=0):
    for a in dir(module):
        if not a.startswith('_'):
            yield getattr(module, a)
    if os.path.isdir(module_path(module)):
        if max_levels > 0:
            for submodule in submodules(module):
                yield from objects_of_module(submodule, max_levels - 1)


def obj_to_dotpath(obj):
    return f"{obj.__module__}.{obj.__name__}"


def finding_objects_of_module_with_given_methods(module, method_names=None, max_levels=1):
    module_dotpath = module.__name__

    objects = {obj_to_dotpath(obj): obj for obj in
               filter(lambda o: isinstance(o, type) and o.__module__.startswith(module_dotpath),
                      objects_of_module(module, max_levels))}

    if method_names is None:
        return objects
    else:
        if isinstance(method_names, str):
            method_names = [method_names]
        method_names = set(method_names)

        return {dotpath: obj for dotpath, obj in objects.items() if method_names.issubset(dir(obj))}


# TODO: Can do a lot more with this (or such) a function:
#  For example, try to return the dotpath that would ACTUALLY correspond to the filepath, given the sys.path
def filepath_to_dotpath(filepath, pkg_paths=None):
    """Figures out a module dotpath from a filepath.
    Checks the sys.path, removing the first common prefix if one is found,
    then changes characters of the path components to make them valid identifier strings"""
    import re

    non_identifier_char_pattern = re.compile(r"\W|^(?=\d)")

    if filepath.endswith('.py'):
        filepath = filepath[:(-len('.py'))]
    elif os.path.isdir(filepath):
        if filepath.endswith(os.path.sep):
            filepath = filepath[:-1]
        if not os.path.isfile(os.path.join(filepath, '__init__.py')):
            raise FileNotFoundError(f"You specified a directory, but the __init__.py file wasn't found in {filepath}")

    if pkg_paths is None:
        pkg_paths = sys.path  # FIXME: TODO: This actually doesn't work as expected

    for path in pkg_paths:
        if filepath.startswith(path):
            filepath = filepath[len(path):]
            break

        print(filepath)
        dotpath = '.'.join((non_identifier_char_pattern.sub('_', x) for x in filepath.split(os.path.sep)))
        if dotpath.startswith('.'):
            dotpath = dotpath[1:]
        return dotpath


def loaded_module_from_filepath(filepath, pkg_paths=None):
    """Get module object from file path (resolving the dotpath automatically)
    """

    dotpath = filepath_to_dotpath(filepath, pkg_paths)
    module_spec = importlib.util.spec_from_file_location(dotpath, filepath)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


# FOLDERPATH->FILEPATH ?->? PATH->DOTPATH->LOADED

# TODO: Complete list
coercion_func = {
    FOLDERPATH: {
        FILEPATH: lambda x: os.path.join(x, '__init__.py')
    },
    # TODO: How do we do FILEPATH to PATH or DOTPATH? With sys.path?
    PATH: {
        DOTPATH: lambda x: '.'.join(x),
    },
    DOTPATH: {
        LOADED: lambda x: importlib.import_module(x)
    }
}


def coerce_module_spec(module_spec,
                       target: ModuleSpecKind = ModuleSpecKind.DOTPATH,
                       source_kind: ModuleSpecKind = None):
    raise NotImplementedError("Not finished")
    source_kind = source_kind or module_spec_kind(module_spec)
    if source_kind == ModuleSpecKind.FOLDERPATH:
        if target == ModuleSpecKind.FILEPATH:
            return os.path.join(module_spec, '__init__.py')
        module_spec = os.path.join(module_spec, '__init__.py')
        source_kind = ModuleSpecKind.FILEPATH
    if source_kind == ModuleSpecKind.FILEPATH:
        module_spec = os.path.join(module_spec, '__init__.py')
        source_kind = ModuleSpecKind.FILEPATH
    return module_spec, source_kind


def get_imported_module_paths(module, recursive_levels=0):
    if inspect.ismodule(module):
        module = module.__file__

    with open(module) as fp:
        root = ast.parse(fp.read(), module)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            imported_module = []
        elif isinstance(node, ast.ImportFrom):
            imported_module = node.module.split('.')
        else:
            continue

        for n in node.names:
            dotpath = '.'.join(imported_module + n.name.split('.'))
            if is_module_dotpath(dotpath):
                yield dotpath
            else:
                dotpath = '.'.join(imported_module)
                if is_module_dotpath(dotpath):
                    yield dotpath

        if recursive_levels > 0:
            yield from get_imported_module_paths

#
# @dataclass
# class Import:
#     module_path: list
#     name: list
#     kind: str
#     alias: str = None
#
#     @classmethod
#     def from_ast_node(cls, node):
#         if isinstance(node, ast.Import):
#             module = []
#         elif isinstance(node, ast.ImportFrom):
#             module = node.module.split('.')
#         else:
#             return None
#
#     @property
#     def module_dotpath(self):
#         return '.'.join(self.module_path)
#
#     @property
#     def module_obj(self):
#         return importlib.import_module(self.module_dotpath)
#
#
# Import = namedtuple("Import", ["module", "name", "alias", "kind"])
