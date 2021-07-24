"""populating projects automatically"""
import os
import subprocess
from wads.populate import populate_pkg_dir, wads_configs
from wads.util import git
from ut.util.context_managers import cd

name_for_url_root = {
    "https://github.com/i2mint": "i2mint",
    "https://github.com/otosense": "otosense",
    "https://github.com/thorwhalen": "thor",
}

proj_root_dir_for_name = {
    "i2mint": "i",
    "otosense": "po",
    "thor": "t",
}

DFLT_PROJ_ROOT_ENVVAR = "PPPP"


def clog(*args, condition=True, log_func=print, **kwargs):
    if condition:
        return log_func(*args, **kwargs)


from functools import partial


def _ensure_no_slash_suffix(string: str) -> str:
    if string.endswith("/"):
        string = string[:-1]
    return string


def _get_org_slash_proj(repo: str) -> str:
    """Gets an org/proj_name string from a url (assuming it's at the end)

    >>> _get_org_slash_proj('https://github.com/thorwhalen/ut/')
    'thorwhalen/ut'
    """
    *_, org, proj_name = _ensure_no_slash_suffix(repo).split("/")
    return f"{org}/{proj_name}"


def _mk_default_project_description(org_slash_proj: str) -> str:
    org, proj_name = org_slash_proj.split("/")
    return f"{proj_name} should say it all, no?"


def get_github_project_description(
    repo: str, default_factory=_mk_default_project_description
):
    """Get project description from github repository, or default if not found"""
    import requests

    org_slash_proj = _get_org_slash_proj(repo)
    api_url = f"https://api.github.com/repos/{org_slash_proj}"
    r = requests.get(api_url)
    if r.status_code == 200:
        description = r.json().get("description", None)
        if description:
            return description
        else:
            return default_factory(org_slash_proj)
    else:
        raise RuntimeError(f"Request response status wasn't 200. Was {r.status_code}")


def populate_proj_from_url(
    url, proj_rootdir=None, description=None, license="apache-2.0", **kwargs
):
    """git clone a repository and set the resulting folder up for packaging."""
    verbose = kwargs.get("verbose", True)
    _clog = partial(clog, condition=verbose)

    url = _ensure_no_slash_suffix(url)

    proj_rootdir = proj_rootdir or os.environ.get(DFLT_PROJ_ROOT_ENVVAR, None)
    assert isinstance(proj_rootdir, str)

    root_url, proj_name = os.path.dirname(url), os.path.basename(url)
    if description is None:
        description = get_github_project_description(url)
    url_name = name_for_url_root.get(root_url, None)
    if url_name:
        _clog(f"url_name={url_name}")

    if url_name is not None and url_name in proj_root_dir_for_name:
        proj_rootdir = os.path.join(proj_rootdir, proj_root_dir_for_name[url_name])
    _clog(f"proj_rootdir={proj_rootdir}")

    with cd(proj_rootdir):
        _clog(f"cloning {url}...")
        subprocess.check_output(f"git clone {url}", shell=True).decode()
        _clog(f"populating package folder...")
        populate_pkg_dir(
            os.path.join(proj_rootdir, proj_name),
            defaults_from=url_name,
            description=description,
            license=license,
            **kwargs,
        )


def gen_key_content_and_description(
    source, excluded_descriptions=(), excluded_keys=(), verbose=True
):
    """Generate key, description, contents triples from a source mapping of python modules (the contents).

    description is extracted from module header, if it exists.
    """
    from tec import file_contents_to_short_description

    excluded_descriptions = set(excluded_descriptions)
    excluded_keys = set(excluded_keys)
    for i, (k, v) in enumerate(source.items()):
        if k not in excluded_keys:
            d = file_contents_to_short_description(v)
            if d:
                if d in excluded_descriptions:
                    if verbose:
                        print(f"Skipping {i}th key ({k}) since description excluded: {d}")
                else:
                    yield k, d, v


class mk_key_to_pkg_name_from_collection:
    """Make a callable that will assign unique package names to keys"""

    def __init__(self, collection_of_pkgs):
        self.collection_of_pkgs = tuple(collection_of_pkgs)
        self.collection_size = len(self.collection_of_pkgs)
        assert (
            len(set(self.collection_of_pkgs)) == self.collection_size
        ), "The items of collection_of_pkgs should be unique"
        self.cursor = 0
        self.assigned = dict()

    def __call__(self, k):
        if k not in self.assigned:
            if self.cursor >= self.collection_size:
                raise IndexError(
                    "I already assigned all collection_of_pkgs items (see them in self.assigned), "
                    "so can't assign {k}"
                )
            self.assigned[k] = self.collection_of_pkgs[self.cursor]
            self.cursor += 1
        return self.assigned[k]


def mk_and_deploy_new_pkgs_from_existing_modules(
    source,
    new_pkg_dir,
    key_to_pkg_name,
    root_url,
    excluded_descriptions=(),
    excluded_keys=(),
    twine_upload_options_str=None,
    target_module_name="__init__.py",
    sleep_seconds_between_deployments=30,
    verbose=True,
):
    """Copy individual modules to a single module new project, and package and deploy it (to pypi)"""
    from typing import Iterable, Callable
    from wads.pack import go
    from ut import print_progress
    from time import sleep
    from py2store import QuickTextStore

    if not isinstance(key_to_pkg_name, Callable) and isinstance(
        key_to_pkg_name, Iterable
    ):
        key_to_pkg_name = mk_key_to_pkg_name_from_collection(key_to_pkg_name)

    target_store = QuickTextStore(new_pkg_dir)

    for k, description, v in gen_key_content_and_description(
        source, excluded_descriptions, excluded_keys, verbose
    ):
        if k not in excluded_keys:
            try:
                without_comma = description.split(",")[0]
                if verbose:
                    print(f"===== {k}: {without_comma} =====")
                pkg_name = key_to_pkg_name(k)
                pkg_dir = os.path.join(new_pkg_dir, pkg_name)
                #         if os.path.isdir(pkg_dir):
                #             raise ValueError("Safety measure")

                target_store[
                    os.path.join(pkg_name, pkg_name, target_module_name)
                ] = v  # copy contents to init module

                populate_pkg_dir(
                    pkg_dir,
                    description=without_comma,
                    author="Thor Whalen",
                    license="apache-2.0",
                    root_url=root_url,
                    url=os.path.join(root_url, k),
                )

                go(
                    pkg_dir,
                    skip_git_commit=True,
                    version="0.0.1",
                    twine_upload_options_str=twine_upload_options_str,
                )
                if verbose:
                    print_progress(f"Sleeping for a bit")
                sleep(sleep_seconds_between_deployments)
            except Exception as e:
                print("!!!!!!!!!!!!!!!!!!")
                print(f"An error occurred: {type(e).__name__}: {e}")
                print("!!!!!!!!!!!!!!!!!!")
