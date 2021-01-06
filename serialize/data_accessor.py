"""Accessing data in a consistent manner"""
__author__ = 'thorwhalen'


import ut.pfile.name as pfile_name
# from ut.khan_utils.encoding import to_unicode_or_bust
from ut.serialize.local import Local
from ut.serialize.s3 import S3


class DataAccessor(object):

    LOCATION_LOCAL = 'LOCAL'
    LOCATION_S3 = 'S3'

    def __init__(self, relative_root=None, mother_root=None, extension=None, force_extension=False, encoding='UTF-8', location=LOCATION_LOCAL, **kwargs):

        # remember these so can switch between locations without having to reset
        self.relative_root = relative_root
        self.extension = extension or ''
        self.force_extension = force_extension or False
        self.encoding = encoding
        self.mother_root = mother_root

        if location==self.LOCATION_LOCAL:
            self.use_local(**kwargs)
        elif location==self.LOCATION_S3:
            self.use_s3(**kwargs)
        else:
            raise AttributeError("Don't know that location (use 'LOCAL' or 'S3')")

    def use_local(self, relative_root=None, mother_root=None, extension=None, force_extension=None, encoding='UTF-8', **kwargs):

        self.currently_using = self.LOCATION_LOCAL

        # replace self vars with any new vals passed in by client
        self.relative_root = relative_root or self.relative_root
        self.extension = extension or self.extension
        self.force_extension = force_extension or self.force_extension
        self.encoding = encoding or 'UTF-8'

        # TODO - we need to update local to accept mother_root argument
        self.local = Local(relative_root=self.relative_root, extension=self.extension, force_extension=self.force_extension, encoding=self.encoding, **kwargs)
        self.dacc = self.local

    def use_s3(self, relative_root=None, mother_root=None, extension=None, force_extension=None, encoding=None, **kwargs):

        self.currently_using = self.LOCATION_S3

        # replace self vars with any new vals passed in by client
        self.relative_root = relative_root or self.relative_root
        self.extension = extension or self.extension
        self.force_extension = force_extension or self.force_extension
        self.encoding = encoding or self.encoding
        self.mother_root = mother_root or self.mother_root
        if self.mother_root is None:
            if self.relative_root.startswith('/'):
                self.relative_root = self.relative_root[1:]
            self.mother_root = pfile_name.get_highest_level_folder(self.relative_root)
            self.relative_root = self.relative_root.replace(self.mother_root,'',1)

        # set folder and other props on s3 via constructor or other
        self.s3 = S3(self.mother_root, self.relative_root, self.extension, self.force_extension, self.encoding, **kwargs)
        self.dacc = self.s3


    def load_excel(self, filename):
        data = self.dacc.loado


    ####################################################################################################################

    def __getattr__(self, name):

        def _missing(*args, **kwargs):

            if self.currently_using == self.LOCATION_LOCAL:
                target = self.local
            elif self.currently_using == self.LOCATION_S3:
                target = self.s3
            else:
                raise RuntimeError("Need to implement method missing for {}.".format(self.currently_using))

            if name in dir(target):
                return getattr(target, name)(*args, **kwargs)
            else:
                raise AttributeError("{} does not exist in {}".format(name, target))

        return _missing



    #
    # def dumpo(self, obj, filename):
    #     """
    #     """
    #
    #     self.dacc.dumpo(obj, filename)
    #     # pickle.dump(obj,open(self.filepath(filename),'w'))
    #
    # def loado(self, filename):
    #     """
    #     loads an object from a local location
    #     """
    #     self.dacc.loado(filename)
    #
    #
    # def dumps(self, the_str, filename, encoding=None):
    #     """
    #     """
    #     if encoding:
    #         s = encoding(the_str)
    #
    #     self.
    #
    #     if self.location == 'local':
    #         self.dacc.dumps(the_str, filename, encoding=encoding)
    #     elif self.location == 's3':
    #         self.dacc.dumps(the_str, filename)
    #
    #
    # def loads(self, filename):
    #     """
    #     loads an object from a local location
    #     """
    #     self.dacc.loads(filename)
    #
    #
    # def dumpu(self, the_str, filename, encoding=None):
    #     """
    #     saves an object to a local location
    #     """
    #     self.dacc.dumps(to_unicode_or_bust(the_str), filename, encoding=encoding)
    #
    #
    # def loadu(self, filename):
    #     """
    #     loads an object from a local location
    #     """
    #     return to_unicode_or_bust(self.dacc.loads(self.filepath(filename)))
    #
    #
    # def df_to_csv(self, df, filename, encoding=None):
    #     """
    #     saves an object to a local location
    #     """
    #     encoding = encoding or self.encoding
    #     return df.to_csv(self.filepath(filename), encoding=encoding, sep="\t")
    #
    #
    # def df_to_excel(self, df, filename):
    #     """
    #     saves an object to a local location
    #     """
    #     if filename.startswith('/'):
    #         filename = filename[1:]
    #     return df.to_excel(self.filepath(filename))

