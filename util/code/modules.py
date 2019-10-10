import os
import inspect
import importlib
import importlib.util
from dataclasses import dataclass
import ast
from enum import Enum
from collections import namedtuple


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
