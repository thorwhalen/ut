__author__ = 'thor'

import pandas as pd
from ut.semantics.termstats import TermStats

dict1 = {'a': 1, 'bb': 2, 'ccc': 3, 'dddd': 4}
dict2 = {'ccc': 30, 'dddd': 40, 'eeeee': 50, 'ffffff': 60}

ts1 = TermStats.from_dict(dict1, name='ts1')
assert all(
    ts1.sr.values
    == pd.Series(data=list(dict1.values()), index=list(dict1.keys()), name='ts1').values
)

print('\n---- ts1')
print(ts1)


ts2 = TermStats.from_dict(dict2, name='ts2')
print('\n---- ts2')
print(ts2)

ts1_times_ts2 = ts1 * ts2
print('\n---- ts1_times_ts2')
print(ts1_times_ts2)

ts1_plus_ts2 = ts1 + ts2
print('\n---- ts1_plus_ts2')
print(ts1_plus_ts2)


print('Trying to get a termstats from www.thorwhalen.com')
import requests

html_ts = TermStats.from_html(requests.get('http://www.thorwhalen.com').text)
print(html_ts.sort().head())
print('')
print(html_ts.sort().tail())
