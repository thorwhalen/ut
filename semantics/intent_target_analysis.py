__author__ = 'thorwhalen'

import pandas as pd
import numpy as np
import semantics.term_stats as ts

class IntentTargetAnalysis(object):
    def __init__(self,
                 get_intent_termstats=None,
                 get_target_termstats=None,
                 similarity_measure=ts.cosine,
                 **kwargs):
        self.get_intent_termstats = get_intent_termstats
        self.get_target_termstats = get_target_termstats
        self.similarity_measure = similarity_measure
        for k,v in list(kwargs.items()):
            self.__setattr__(k,v)

    def intent_target_similarity(self, intent, target):
        return self.similarity_measure(
            self.get_intent_termstats(intent),
            self.get_target_termstats(target)
        )

    def match_target_thru_progressive_terms(self, intent):
        intent_ts = self.get_intent_termstats(intent)
        intent_ts = intent_ts.sort(columns=['stat'], ascending=False)


    def close_stores(self):
        attrs = self.__dict__
        for k,v in list(attrs.items()):
            if isinstance(attrs[k], pd.HDFStore):
                print("closing %s" % k)
                attrs[k].close()
