__author__ = 'thor'

import ut.daf.ch as daf_ch
import ut.pdict.to as pdict_to

awe_col_synonyms = {
    'Campaign': ['campaign'],
    'Ad Group': ['ad_group', 'ad_group_name'],
    'Keyword': ['keyword', 'keyword_text'],
    'Max. CPC': ['max_cpc'],
    'match type': ['match_type'],
    'Destination URL': ['destination_url']
}

awe_col_replacer = pdict_to.word_replacer(pdict_to.inverse_one_to_many(awe_col_synonyms))


def mk_awe_cols(df):
    old_cols = df.columns
    new_cols = list(map(awe_col_replacer, old_cols))
    return daf_ch.ch_col_names(df, new_cols, old_cols)

