__author__ = 'thorwhalen'

#import daf.get as dfg
import nltk
import pandas as pd

# input: data_folder='', dataname, savename='',
# output: string of ascii char correspondents
#   (replacing, for example, accentuated letters with non-accentuated versions of the latter)
def mk_termCounts(dat,indexColName,strColName,data_folder=''):
    from util import log
    from daf.get import get_data
    dat = get_data(dat,data_folder)
    log.printProgress("making {} word counts (wc)",strColName)
    sr = to_kw_fd(dat[strColName])
    sr.index = dat.hotel_id.tolist()
    return sr
    # translate fds to series
    #printProgress("translating {} FreqDist to Series",col)
    #sr = fd_to_series(sr)
    #printProgress("saving {} word counts (wc)",col)
    #save_data(sr,savename + col)
    #printProgress('Done!')

# input: daf of strings
# output: series of the tokens of the strings, processed for AdWords keywords
#   i.e. string is lower capsed and asciied, and words are [\w&]+
def to_kw_tokens(dat,indexColName,strColName,data_folder=''):
    from util import log
    import pstr.trans
    from daf.get import get_data
    dat = get_data(dat,data_folder)
    log.printProgress("making {} tokens",strColName)
    sr = dat[strColName]
    sr.index = dat.hotel_id.tolist()
    # preprocess string
    sr = sr.map(lambda x: x.lower())
    sr = sr.map(lambda x: pstr.trans.toascii(x))
    # tokenize
    sr = sr.map(lambda x:nltk.regexp_tokenize(x,'[\w&]+'))
    # return this
    return sr

# input: string, series (with string columns)
# output: nltk.FreqDist (word count) of the tokens of the string, processed for AdWords keywords
#   i.e. string is lower capsed and asciied, and words are [\w&]+
def to_kw_fd(s):
    import pstr.trans
    if isinstance(s,pd.Series):
        # preprocess string
        s = s.map(lambda x: x.lower())
        s = s.map(lambda x: pstr.trans.toascii(x))
        # tokenize
        s = s.map(lambda x:nltk.regexp_tokenize(x,'[\w&]+'))
        # return series of fds
        return s.map(lambda tokens:nltk.FreqDist(tokens))
    elif isinstance(s,str):
        # preprocess string
        s = pstr.trans.toascii(s.lower())
        # tokenize
        tokens = nltk.regexp_tokenize(s,'[\w&]+')
        return nltk.FreqDist(tokens)

# input: nltk.FreqDist, pd.Series (with FreqDist in them), or list of FreqDists
# output: the pd.Series representation of the FreqDist
def fd_to_series(fd):
    if isinstance(fd,nltk.FreqDist):
        return pd.Series(fd.values(),fd.keys())
    elif isinstance(fd,pd.Series):
        return fd.map(lambda x:pd.Series(x.values(),x.keys()))
    elif isinstance(fd,list):
        return map(lambda x:pd.Series(x.values(),x.keys()))