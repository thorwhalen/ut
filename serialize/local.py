__author__ = 'thorwhalen'

from ut.datapath import datapath
import pickle
import os
from ut.util.importing import get_environment_variable
import pandas as pd
import ut.pfile.to as file_to
import ut.pfile.name as pfile_name
import ut.pstr.to as str_to
from ut.pstr.trans import str_to_unicode_or_bust
#from os import environ # does this load the whole array? Can we just take MS_DATA instead?

try:
    MS_DATA = get_environment_variable('MS_DATA')
except KeyError:
    MS_DATA = ''

ENCODING_NONE = 0
ENCODING_UNICODE = 1
ENCODING_UTF8 = 2


class Local(object):

    def __init__(self, relative_root=None, extension='', force_extension=False, encoding='UTF-8', mother_root=MS_DATA):
        if relative_root:
            self.root_folder = os.path.join(mother_root, relative_root)
        else:
            self.root_folder = mother_root
        self.extension = extension
        self.force_extension = force_extension
        self.encoding = encoding


    def filepath(self, filename=None):

        if not filename:
            return self.root_folder

        # remove slash suffix if present (because self.sound_file_root_folder ends with / already)
        if filename.startswith('/'):
            filename = filename[1:]
        # add extension
        if self.extension:
            if self.force_extension:
                filename = pfile_name.replace_extension(filename, self.extension)
            else:
                filename = pfile_name.add_extension_if_not_present(filename, self.extension)
        return os.path.join(self.root_folder, filename)


    def dumpo(self, obj, filename):
        """
        saves an object to a local location
        """
        pickle.dump(obj,open(self.filepath(filename),'w'))

    def loado(self, filename):
        """
        loads an object from a local location
        """
        return pickle.load(open(self.filepath(filename),'r'))


    def dumps(self, the_str, filename, encoding=None):
        """
        saves an object to a local location
        """
        encoding = encoding or self.encoding
        str_to.file(the_str, self.filepath(filename), encoding=encoding)


    def loads(self, filename):
        """
        loads an object from a local location
        """
        return file_to.string(self.filepath(filename))


    def dumpu(self, the_str, filename, encoding=None):
        """
        saves an object to a local location
        """
        encoding = encoding or self.encoding
        str_to.file(str_to_unicode_or_bust(the_str), self.filepath(filename), encoding=encoding)


    def loadu(self, filename):
        """
        loads an object from a local location
        """
        return str_to_unicode_or_bust(file_to.string(self.filepath(filename)))


    def df_to_csv(self, df, filename, encoding=None):
        """
        saves an object to a local location
        """
        encoding = encoding or self.encoding
        return df.to_csv(self.filepath(filename), encoding=encoding, sep="\t")


    def df_to_excel(self, df, filename, **kwargs):
        """
        saves an object to a local location
        """
        if filename.startswith('/'):
            filename = filename[1:]
        if not kwargs.has_key('index'): # make the default index=False if not specified in input
            kwargs = dict({'index': False}, **kwargs)
        return df.to_excel(self.filepath(filename), **kwargs)

    def excel_to_df(self, filename, sheetname=None, **kwargs):
        """
        returns a df from the excel file specified by filename, taking the sheet specified
        by sheetname (which should be the sheetname, but could also be the sheet number (starting by 0)). Default
        sheetname is the first sheet.
        Further kwargs can be passed to pd.ExcelFile.parse
        """
        xd = pd.ExcelFile(self.filepath(filename))
        if sheetname is None:
            sheetname = xd.sheet_names[0]
            print "No sheetname specified so I'm taking the first one: %s" % sheetname
        elif isinstance(sheetname, int):
            sheetname = xd.sheet_names[sheetname]
        return xd.parse(sheetname, **kwargs)