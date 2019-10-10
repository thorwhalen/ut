# Author: Tom Burge
# Contributor: Anthon van der Neut - Thanks for the help!
# Title: py-yaml-json-parser
# Description: Simple Menu-Based YAML/JSON Parser
# Requirements:
#   - Ruamel.YAML (pip install ruamel.yaml
#
# See: https://stackoverflow.com/questions/51914505/python-yaml-to-json-to-yaml for explanations

import json
import sys
from collections.abc import Mapping, Sequence
from collections import OrderedDict
import ruamel.yaml
from ruamel.yaml.error import YAMLError
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import PreservedScalarString, SingleQuotedScalarString
from ruamel.yaml.compat import string_types, MutableMapping, MutableSequence

yaml = ruamel.yaml.YAML()


class OrderlyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Mapping):
            return OrderedDict(o)
        elif isinstance(o, Sequence):
            return list(o)
        return json.JSONEncoder.default(self, o)


def printmenu():
    menu = ('1. Parse YAML to JSON\n'
            '2. Parse JSON to YAML\n'
            '3. Exit\n'
            )
    print(menu)


def preserve_literal(s):
    return PreservedScalarString(s.replace('\r\n', '\n').replace('\r', '\n'))


def walk_tree(base):
    if isinstance(base, MutableMapping):
        for k in base:
            v = base[k]  # type: Text
            if isinstance(v, string_types):
                if '\n' in v:
                    base[k] = preserve_literal(v)
                elif '${' in v or ':' in v:
                    base[k] = SingleQuotedScalarString(v)
            else:
                walk_tree(v)
    elif isinstance(base, MutableSequence):
        for idx, elem in enumerate(base):
            if isinstance(elem, string_types):
                if '\n' in elem:
                    base[idx] = preserve_literal(elem)
                elif '${' in elem or ':' in elem:
                    base[idx] = SingleQuotedScalarString(elem)
            else:
                walk_tree(elem)

def parseyaml(intype, outtype):
    infile = input('Please enter a {} filename to parse: '.format(intype))
    outfile = input('Please enter a {} filename to output: '.format(outtype))

    with open(infile, 'r') as stream:
        try:
            datamap = yaml.load(stream)
            with open(outfile, 'w') as output:
                output.write(OrderlyJSONEncoder(indent=2).encode(datamap))
        except YAMLError as exc:
            print(exc)
            return False
    print('Your file has been parsed.\n\n')


def parsejson(intype, outtype):
    infile = input('Please enter a {} filename to parse: '.format(intype))
    outfile = input('Please enter a {} filename to output: '.format(outtype))

    with open(infile, 'r') as stream:
        try:
            datamap = json.load(stream, object_pairs_hook=CommentedMap)
            walk_tree(datamap)
            with open(outfile, 'w') as output:
                yaml.dump(datamap, output)
        except YAMLError as exc:
            print(exc)
            return False
    print('Your file has been parsed.\n\n')


loop = True

while loop:
    printmenu()  # Prints Menu for User
    choice = int(input('Please select an operation: '))

    if choice == 1:
        infiletype = 'YAML'
        outfiletype = 'JSON'
        parseyaml(infiletype, outfiletype)
    elif choice == 2:
        infiletype = 'JSON'
        outfiletype = 'YAML'
        parsejson(infiletype, outfiletype)
    elif choice == 3:
        sys.exit(0)
    else:
        print('Please make a selection')

