#! /usr/bin/env python
# coding: utf-8

# Author: João S. O. Bueno
# Copyright (c) 2009 - Fundação CPqD
# License: LGPL V3.0


from types import ModuleType, FunctionType, ClassType
import sys
import imp


def find_dependent_modules():
    """gets a one level inversed module dependence tree"""
    tree = {}
    for module in list(sys.modules.values()):
        if module is None:
            continue
        tree[module] = set()
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, ModuleType):
                tree[module].add(attr)
            elif type(attr) in (FunctionType, ClassType):        
                tree[module].add(attr.__module__)
    return tree


def get_reversed_first_level_tree(tree):
    """Creates a one level deep straight dependence tree"""
    new_tree = {}
    for module, dependencies in list(tree.items()):
        for dep_module in dependencies:
            if dep_module is module:
                continue
            if not dep_module in new_tree:
                new_tree[dep_module] = set([module])
            else:
                new_tree[dep_module].add(module)
    return new_tree


def find_dependants_recurse(key, rev_tree, previous=None):
    """Given a one-level dependance tree dictionary,
       recursively builds a non-repeating list of all dependant
       modules
    """
    if previous is None:
        previous = set()
    if not key in rev_tree:
        return []
    this_level_dependants = set(rev_tree[key])
    next_level_dependants = set()
    for dependant in this_level_dependants:
        if dependant in previous:
            continue
        tmp_previous = previous.copy()
        tmp_previous.add(dependant)
        next_level_dependants.update(
             find_dependants_recurse(dependant, rev_tree,
                                     previous=tmp_previous,
                                    ))
    # ensures reloading order on the final list
    # by postponing the reload of modules in this level
    # that also appear later on the tree
    dependants = (list(this_level_dependants.difference(
                        next_level_dependants)) +
                  list(next_level_dependants))
    return dependants


def get_reversed_tree():
    """
        Yields a dictionary mapping all loaded modules to
        lists of the tree of modules that depend on it, in an order
        that can be used fore reloading
    """
    tree = find_dependent_modules()
    rev_tree = get_reversed_first_level_tree(tree)
    compl_tree = {}
    for module, dependant_modules in list(rev_tree.items()):
        compl_tree[module] = find_dependants_recurse(module, rev_tree)
    return compl_tree

def reload_dependences(module):
    """
        reloads given module and all modules that
        depend on it, directly and otherwise.
    """
    tree = get_reversed_tree()
    imp.reload(module)
    for dependant in tree[module]:
        imp.reload(dependant)