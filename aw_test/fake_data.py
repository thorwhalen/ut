__author__ = 'thorwhalen'
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df9_for_dedup = pd.DataFrame(
    {'keyword': ['rome hotel','Rome Hotel', 'rome hôtel', ' rome * **  hotel $ ', 'hotel rome', 'rome hotel', 'hotel rome', 'hôtel rome', 'rome hotel', 'rome hotel', 'matt monkey'],
     'match_type':          ['Broad',      'Broad',     'Broad',  'Broad',       'Broad',      'Exact',      'Exact',     'Exact',      'Phrase',      'Broad',      'Broad'],
     'ad_group':            ['AG',         'AG',        'AG',   'AG',          'AG',         'AG',         'AG',        'AG',         'AG',          'AG_2',       'AG_2'],
     'campaign':            ['CP',         'CP',        'CP',  'CP',          'CP',         'CP',         'CP',        'CP',         'CP',          'CP',         'CP_X']},
    columns=['keyword','match_type','ad_group','campaign'])


df9_for_russian_dolls = pd.DataFrame(
    {'keyword': ['+rome +hotel','rome hotel', 'rome hotel', 'paris hotel', 'paris hotel', '+paris +resort', 'paris resort', 'paris hilton', '+milan +b&b', 'milan b&b'],
     'match_type': ['Broad',     'Phrase',    'Exact',      'Phrase',      'Exact',       'Broad',          'Phrase',       'Exact',        'Broad',     'Exact'],
     'max_cpc': [   3,         2,        1,                 10,          20,             100,              100,             0.1,           2000,          1000],
     'ad_group':  ['Rome',       'Rome',      'Rome',       'Paris',       'Paris',       'Paris',         'Paris',         'Paris',        'Milan',     'Milan'],
     'campaign': ['CP',         'CP',        'CP',         'CP',          'CP',         'CP',         'CP',                 'CP',         'CP',          'CP']},
    columns=['keyword','match_type','max_cpc','ad_group','campaign'])