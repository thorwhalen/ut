"""
A few functions to investigate what objects can be imported from a module
(and the depth of the dot-path to import those objects directly).

The main function, print_top_level_diagnosis,
prints a diagnosis of the imports that can be optained from the (top level) module.
That is, those objects that can by imported by doing:
```
from module import obj
```
though the object's code may be several package levels down (say module.sub1.sub2.obj).


```
>> import numpy, pandas, scipy
>> print_top_level_diagnosis(numpy)
--------- numpy ---------
601 objects can be imported from top level numpy:
  20 modules
  300 functions
  104 types

depth	count
0	163
1	406
2	2
3	29
4	1

>> print_top_level_diagnosis(pandas)
--------- pandas ---------
115 objects can be imported from top level pandas:
  12 modules
  55 functions
  40 types

depth	count
0	12
3	37
4	65
5	1

>> print_top_level_diagnosis(scipy)
--------- scipy ---------
582 objects can be imported from top level scipy:
  9 modules
  412 functions
  96 types

depth	count
0	61
1	395
2	4
3	122
```
"""

from pkg_resources import get_distribution, DistributionNotFound, RequirementParseError
from importlib import import_module


def read_requirements(requirements_file):
    with open(requirements_file, 'r') as f:
        return f.read().splitlines()


def get_module_name(package, on_error='raise'):
    try:
        t = list(get_distribution(package)._get_metadata('top_level.txt'))
        if t:
            return t[0]
        else:
            return None
    except Exception as e:
        if on_error == 'raise':
            raise
        elif on_error == 'error_class':
            return e.__class__.__name__
        else:
            return on_error  # just the value specified by on_error


from types import ModuleType, FunctionType

take_everything = lambda x: True


def top_level_objs(module, obj_filt=take_everything):
    top_level_imports = [x for x in dir(module) if not x.startswith('_')]
    for a in top_level_imports:
        obj = getattr(module, a)
        if obj_filt(obj):
            yield obj


def obj_module_depth_counts(module, obj_filt=take_everything):
    from collections import Counter
    def depth(obj):
        if not hasattr(obj, '__module__'):
            return 0
        else:
            return len(obj.__module__.split('.'))

    return sorted(Counter(depth(obj) for obj in top_level_objs(module, obj_filt)).items())


def print_top_level_diagnosis(module, obj_filt=take_everything):
    """
    Prints a diagnosis of the imports that can be optained from the (top level) module.
    That is, those objects that can by imported by doing:
    ```
    from module import obj
    ```
    though the object's code may be several package levels down (say module.sub1.sub2.obj).

    :param module: The module (package) to analyze
    :param obj_filt: The filter to apply (to the objects)

    ```
    >> import numpy, pandas, scipy
    >> print_top_level_diagnosis(numpy)
    --------- numpy ---------
    601 objects can be imported from top level numpy:
      20 modules
      300 functions
      104 types

    depth	count
    0	163
    1	406
    2	2
    3	29
    4	1

    >> print_top_level_diagnosis(pandas)
    --------- pandas ---------
    115 objects can be imported from top level pandas:
      12 modules
      55 functions
      40 types

    depth	count
    0	12
    3	37
    4	65
    5	1

    >> print_top_level_diagnosis(scipy)
    --------- scipy ---------
    582 objects can be imported from top level scipy:
      9 modules
      412 functions
      96 types

    depth	count
    0	61
    1	395
    2	4
    3	122
    ```
    """
    if isinstance(module, str):
        module = import_module(module)
    print(f"\n--------- {module.__name__} ---------")
    print(f"{len(list(top_level_objs(module)))} objects can be imported from top level {module.__name__}:")
    for kind in [ModuleType, FunctionType, type]:
        print(f"  {len(list(top_level_objs(module, lambda x: isinstance(x, kind))))} {kind.__name__}s")
    print("")
    print(f"depth\tcount")
    for depth, count in obj_module_depth_counts(module, obj_filt):
        print(f'{depth}\t{count}')
    print("")
