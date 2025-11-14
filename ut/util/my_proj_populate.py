"""populating projects automatically"""
import os
import subprocess
from wads.populate import populate_pkg_dir, wads_configs
from wads.util import git
from ut.util.context_managers import cd

from wads.populate import (
    name_for_url_root,
    proj_root_dir_for_name,
    DFLT_PROJ_ROOT_ENVVAR,
    clog,
    get_github_project_description,
    populate_proj_from_url,
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
                        print(
                            f'Skipping {i}th key ({k}) since description excluded: {d}'
                        )
                else:
                    yield k, d, v


class mk_key_to_pkg_name_from_collection:
    """Make a callable that will assign unique package names to keys"""

    def __init__(self, collection_of_pkgs):
        self.collection_of_pkgs = tuple(collection_of_pkgs)
        self.collection_size = len(self.collection_of_pkgs)
        assert (
            len(set(self.collection_of_pkgs)) == self.collection_size
        ), 'The items of collection_of_pkgs should be unique'
        self.cursor = 0
        self.assigned = dict()

    def __call__(self, k):
        if k not in self.assigned:
            if self.cursor >= self.collection_size:
                raise IndexError(
                    'I already assigned all collection_of_pkgs items (see them in self.assigned), '
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
    *,
    excluded_descriptions=(),
    excluded_keys=(),
    twine_upload_options_str=None,
    target_module_name='__init__.py',
    sleep_seconds_between_deployments=30,
    verbose=True,
    author='Thor Whalen',
    license='apache-2.0',
    skip_git_commit=True,
    version='0.0.1',
):
    """Copy individual modules to a single module new project, and package and deploy it (to pypi)"""
    from collections.abc import Iterable, Callable
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
                without_comma = description.split(',')[0]
                if verbose:
                    print(f'===== {k}: {without_comma} =====')
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
                    author=author,
                    license=license,
                    root_url=root_url,
                    url=os.path.join(root_url, k),
                )

                go(
                    pkg_dir,
                    skip_git_commit=skip_git_commit,
                    version=version,
                    twine_upload_options_str=twine_upload_options_str,
                )
                if verbose:
                    print_progress(f'Sleeping for a bit')
                sleep(sleep_seconds_between_deployments)
            except Exception as e:
                print('!!!!!!!!!!!!!!!!!!')
                print(f'An error occurred: {type(e).__name__}: {e}')
                print('!!!!!!!!!!!!!!!!!!')
