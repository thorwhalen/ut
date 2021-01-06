"""populating projects automatically"""
import os
import subprocess
from wads.populate import populate_pkg_dir, wads_configs
from wads.util import git
from ut.util.context_managers import cd

name_for_url_root = {
    'https://github.com/i2mint': 'i2mint',
    'https://github.com/otosense': 'otosense',
    'https://github.com/thorwhalen': 'thor',
}

proj_root_dir_for_name = {
    'i2mint': 'i',
    'otosense': 'po',
    'thor': 't',
}

DFLT_PROJ_ROOT_ENVVAR = 'PPPP'


def clog(*args, condition=True, log_func=print, **kwargs):
    if condition:
        return log_func(*args, **kwargs)


from functools import partial


def populate_proj_from_url(url, proj_rootdir=None,
                           description=None,
                           license='apache-2.0',
                           **kwargs):
    """git clone a repository and set the resulting folder up for packaging.
    """
    verbose = kwargs.get('verbose', True)
    _clog = partial(clog, condition=verbose)

    if url.endswith('/'):
        url = url[:-1]

    proj_rootdir = proj_rootdir or os.environ.get(DFLT_PROJ_ROOT_ENVVAR, None)
    assert isinstance(proj_rootdir, str)

    root_url, proj_name = os.path.dirname(url), os.path.basename(url)
    if description is None:
        description = f"{proj_name} should say it all, no?"
    url_name = name_for_url_root.get(root_url, None)
    if url_name:
        _clog(f"url_name={url_name}")

    if url_name is not None and url_name in proj_root_dir_for_name:
        proj_rootdir = os.path.join(proj_rootdir, proj_root_dir_for_name[url_name])
    _clog(f"proj_rootdir={proj_rootdir}")

    with cd(proj_rootdir):
        _clog(f"cloning {url}...")
        subprocess.check_output(f'git clone {url}', shell=True).decode()
        _clog(f"populating package folder...")
        populate_pkg_dir(os.path.join(proj_rootdir, proj_name),
                         defaults_from=url_name,
                         description=description,
                         license=license,
                         **kwargs)
