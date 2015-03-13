__author__ = 'thorwhalen'

from datapath import datapath
import pickle
import os
import pandas as pd
import pfile.to as file_to
import pfile.name as file_name
import pstr.to as str_to
from khan_utils.encoding import to_unicode_or_bust

class DataAccessor(object):

    def __init__(self, data_root_location=None, extension='', force_extension = False, encoding='UTF-8'):
        self.root_folder = datapath(data_root_location)
        self.extension = extension
        self.force_extension = force_extension
        self.encoding = encoding

    def filepath(self,filename):
        # remove slash suffix if present
        if filename.startswith('/'):
            filename = filename[1:]
        # add extension
        if self.extension:
            if self.force_extension:
                if filename=='':
                    filename = file_name.replace_extension(filename)
                else:
                    filename = file_name.replace_extension(filename,self.extension)
            else:
                if filename=='':
                    filename = file_name.add_extension_if_not_present(filename)
                else:
                    filename = file_name.add_extension_if_not_present(filename,self.extension)
        return os.path.join(self.root_folder,filename)


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


    def dumps(self, the_str, filename, encoding="UTF-8"):
        """
        saves an object to a local location
        """
        str_to.file(the_str, self.filepath(filename), encoding=encoding)

    def loads(self, filename):
        """
        loads an object from a local location
        """
        return file_to.string(self.filepath(filename))

    def dumpu(self, the_str, filename, encoding="UTF-8"):
        """
        saves an object to a local location
        """
        str_to.file(to_unicode_or_bust(the_str), self.filepath(filename), encoding=encoding)

    def loadu(self, filename):
        """
        loads an object from a local location
        """
        return to_unicode_or_bust(file_to.string(self.filepath(filename)))


    def df_to_csv(self, df, filename):
        """
        saves an object to a local location
        """
        return df.to_csv(self.filepath(filename),encoding='UTF-8',sep="\t")

    def df_to_excel(self, df, filename):
        """
        saves an object to a local location
        """
        if filename.startswith('/'):
            filename = filename[1:]
        return df.to_excel(self.filepath(filename))