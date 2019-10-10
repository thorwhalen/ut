__author__ = 'thor'

import numpy as np
import io
import pandas as pd
import itertools
from collections import Counter
from nltk.corpus import wordnet as wn


def print_word_definitions(word):
    print(word_definitions_string(word))


def word_definitions_string(word):
    return '\n'.join(['%d: %s (%s)'
                      % (i, x.definition(), x.name()) for i, x in enumerate(wn.synsets(word))])


def print_word_lemmas(word):
    t = Counter([l.name for s in wn.synsets(word) for l in s.lemmas])
    print(pd.Series(index=list(t.keys()), data=list(t.values())).sort(inplace=False, ascending=False))


def _lemma_names_str(syn):
    return '(' + ', '.join(syn.lemma_names) + ')'


def print_hypos_with_synset(syn, tab=''):
    print(tab + syn.name)
    h = syn.hyponyms()
    if len(h) > 0:
        for hi in h:
            print_hypos_with_synset(hi, tab + '  ')
    else:
        print(tab + '  ' + _lemma_names_str(syn))


def pprint_hypos(syn, tab=''):
    print(tab + _lemma_names_str(syn))
    h = syn.hyponyms()
    if len(h) > 0:
        for hi in h:
            pprint_hypos(hi, tab + '  ')


class iTree(object):
    def __init__(self, value=None):
        self.value = value
        self.children = []
        self.default_node_2_str = lambda node: str(node.value)

    def __iter__(self):
        for v in itertools.chain(*map(iter, self.children)):
            yield v
        yield self

    def tree_info_str(self,
                      node_2_str=None,  # default info is node value
                      tab_str=2 * ' ',  # tab string
                      depth=0
                      ):
        node_2_str = node_2_str or self.default_node_2_str
        s = depth * tab_str + node_2_str(self) + '\n'
        new_depth = depth + 1
        for child in self.children:
            s += child.tree_info_str(node_2_str, tab_str, new_depth)
        return s


class HyponymTree(iTree):
    def __init__(self, value=None):
        if isinstance(value, str):
            value = wn.synset(value)
        super(HyponymTree, self).__init__(value=value)
        for hypo in value.hyponyms():
            self.children.append(HyponymTree(hypo))
        self.set_default_node_2_str('name')

    def __str__(self):
        return self.value.name()

    def __repr__(self):
        return self.value.name()

    def print_lemmas(self, tab=''):
        print(tab + _lemma_names_str(self.value))
        for c in self.children:
            pprint_hypos(c, tab + '  ')

    def leafs(self):
        return [x for x in self]

    @classmethod
    def of_hyponyms(cls, syn):
        tree = cls(syn)
        for hypo in syn.hyponyms():
            tree.children.append(cls.of_hyponyms(hypo))
        return tree

    @staticmethod
    def get_node_2_str_function(method='name', **kwargs):
        """
        returns a node_2_str function (given it's name)
        method could be
            * 'name': The synset name (example sound.n.01)
            * 'lemma_names': A parenthesized list of lemma names
            * 'name_and_def': The synset name and it's definition
            * 'lemmas_and_def': The lemma names and definition
        """
        if method == 'name':
            return lambda node: \
                node.value.name
        elif method == 'lemma_names' or method == 'lemmas':
            lemma_sep = kwargs.get('lemma_sep', ', ')
            return lambda node: \
                '(' + lemma_sep.join(node.value.lemma_names) + ')'
        elif method == 'name_and_def':
            return lambda node: \
                node.value.name + ': ' + node.value.definition
        elif method == 'lemmas_and_def':
            lemma_sep = kwargs.get('lemma_sep', ', ')
            def_sep = kwargs.get('def_sep', ': ')
            return lambda node: \
                '(' + lemma_sep.join(node.value.lemma_names) + ')' \
                + def_sep + node.value.definition
        elif method == 'all':
            lemma_sep = kwargs.get('lemma_sep', ', ')
            def_sep = kwargs.get('def_sep', ': ')
            return lambda node: \
                '(' + lemma_sep.join(node.value.lemma_names) + ')' \
                + def_sep + node.value.name \
                + def_sep + node.value.definition
        else:
            raise ValueError("Unknown node_2_str_function method")

    def set_default_node_2_str(self, method='name'):
        """
        will set the default string representation of a synset
        (used as a default ny the tree_info_str function for example)
        from the name of the method to use
        (see get_node_2_str_function(method))
        method could be
            * 'name': The synset name (example sound.n.01)
            * 'lemma_names': A parenthesized list of lemma names
            * 'name_and_def': The synset name and it's definition
            * 'lemmas_and_def': The lemma names and definition
        """
        self.default_node_2_str = HyponymTree.get_node_2_str_function(method)

    def _df_for_excel_export(self, method='all', method_args={}):
        method_args['def_sep'] = ':'
        method_args['tab_str'] = method_args.get('tab_str', '* ')
        s = ''
        # s = 'lemmas' + method_args['def_sep'] + 'synset' + method_args['def_sep'] + 'definition' + '\n'
        s += self.tree_info_str(node_2_str=self.get_node_2_str_function(method=method),
                                tab_str=method_args['tab_str'])
        return pd.DataFrame.from_csv(io.StringIO(str(s)),
                                     sep=method_args['def_sep'], header=None, index_col=None)

    def export_info_to_excel(self, filepath, sheet_name='hyponyms', method='all', method_args={}):
        d = self._df_for_excel_export(method=method, method_args=method_args)
        d.to_excel(filepath, sheet_name=sheet_name, header=False, index=False)


class HyponymForest(object):
    def __init__(self, tree_list):
        assert len(tree_list) == len(np.unique(tree_list)), "synsets in list must be unique"
        for i, ss in enumerate(tree_list):
            if not isinstance(ss, HyponymTree):
                tree_list[i] = HyponymTree(ss)
        self.tree_list = tree_list

    def leafs(self):
        return np.unique([xx for x in self.tree_list for xx in x.leafs()])

    def export_info_to_excel(self, filepath, sheet_name='hyponyms', method='all', method_args={}):
        d = pd.DataFrame()
        for dd in self.tree_list:
            d = pd.concat([d, dd._df_for_excel_export(method=method, method_args=method_args)])
        d.to_excel(filepath, sheet_name=sheet_name, header=False, index=False)
