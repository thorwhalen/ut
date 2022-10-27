__author__ = 'thor'

import os
from ftplib import FTP

from ut.util.log import printProgress


def get_ftp_files_I_dont_have(
    ftp_kwargs, remote_dir='.', local_dir='.', remote_filename_filter=None
):
    """
    Getting files from a remote ftp folder to a local folder
    """
    ftp = FTP(**ftp_kwargs)

    # getting a (possibly filtered) list of remote files
    ftp_files = ftp.nlst(remote_dir)
    if remote_filename_filter is not None:
        printProgress('Remote files: %d' % len(ftp_files))
        try:
            ftp_files = list(filter(remote_filename_filter, ftp_files))
        except Exception:
            ftp_files = remote_filename_filter(ftp_files)
        printProgress(
            '... After filtering, only %d files left that will be processed'
            % len(ftp_files)
        )
    else:
        printProgress('Remote files: %d' % len(ftp_files))

    list_of_files_to_fetch = []
    for f in ftp_files:
        local_filepath = os.path.join(local_dir, f)
        if not os.path.exists(local_filepath):
            list_of_files_to_fetch.append([f])

    # for each of the files that local_directory doesn't have, retrieve it and store it locally
    printProgress(
        "Fetching the %d files that (%s) didn't have"
        % (len(list_of_files_to_fetch), local_dir)
    )
    for f in list_of_files_to_fetch:
        local_filepath = os.path.join(local_dir, f)
        remote_filepath = os.path.join(remote_dir, f)
        printProgress('Getting and storing file to %s' % local_filepath)
        get_file_from_ftp(ftp, remote_filepath, local_filepath)


def get_file_from_ftp(ftp, remote_filepath, local_filepath):
    """
    Getting a single file from a remote ftp server.
    """
    fp = open(local_filepath, 'wb')
    ftp.retrbinary('RETR %s' % remote_filepath, fp.write)
    fp.close()
