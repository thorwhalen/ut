"""utils for ipython notebooks"""

from __future__ import division

import os, json
from ut.pstr.to import file as str_to_file
import re
from ut.pfile.iter import recursive_file_walk_iterator_with_filepath_filter

default_link_root = 'http://localhost:8888/notebooks/'


def _link_root_from_port(port):
    return 'http://localhost:{}/notebooks/'.format(port)


def _mk_link_root(x):
    if isinstance(x, int):
        return _link_root_from_port(x)
    elif x.startswith('http'):
        return x
    else:
        return 'http://' + x


def max_common_prefix(a):
    """
    Given a list of strings, returns the longest common prefix
    :param a: list-like of strings
    :return: the smallest common prefix of all strings in a
    """
    if not a:
        return ''
    # Note: Try to optimize by using a min_max function to give me both in one pass. The current version is still faster
    s1 = min(a)
    s2 = max(a)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def all_table_of_contents_html_from_notebooks(notebooks,
                                              title=None,
                                              link_root=default_link_root,
                                              recursive: bool=False,
                                              save_to_file='table_of_contents.html'):
    """
    Make an html page containing the table of contents of the listed notebooks, with
    links that will open the notebook and bring you to that section.
    Just wow.

    :param notebooks: List of notebook filepaths, or folder that contains notebooks.
    :param title: Title of html page
    :param link_root: Root url to use for links
    :param recursive: Whether to explore subfolders recursively
    :param save_to_file: File where the html should be saved.
    :return:
    """
    folder = None
    if isinstance(notebooks, str) and os.path.isdir(notebooks):
        folder = os.path.abspath(os.path.expanduser(notebooks))
        notebooks = ipynb_filepath_list(folder, recursive=recursive)
        title = title or folder
        s = "<b>{}</b><br><br>\n\n".format(title)
    elif title:
        s = "<b>{}</b><br><br>\n\n".format(title)
    else:
        s = '' \
            ''

    for f in notebooks:
        if folder is not None:
            _link_root = os.path.join(link_root, os.path.dirname(f[(len(folder) + 1):]))
        else:
            _link_root = link_root
        ss = table_of_contents_html_from_notebook(f, link_root=_link_root)
        if ss is not None:
            s += ss + '<br>\n\n'
    if save_to_file is None:
        return s
    else:
        if not isinstance(save_to_file, str):
            save_to_file = 'table_of_contents.html'
        str_to_file(s, save_to_file)


def _mk_link_html(filename, link_root=default_link_root):
    link_root = _mk_link_root(link_root)
    url = os.path.join(link_root, filename)
    if filename.endswith('.ipynb'):
        filename = filename[:-len('.ipynb')]
    return '<b><a href="{}">{}</a></b>'.format(url, filename)


def _append_link_root_to_all_pound_hrefs(html, link_root=default_link_root):
    return re.sub(r'href="(#[^"]+)"',
                  r'href="{}\1"'.format(link_root),
                  html)


def table_of_contents_html_from_notebook(ipynb_filepath,
                                         link_root=default_link_root):
    filename = os.path.basename(ipynb_filepath)
    d = json.load(open(ipynb_filepath)).get('cells', None)
    if d is not None and isinstance(d, list) \
            and len(d) > 0 and 'source' in d[0] \
            and len(d[0]['source']) >= 2:
        if d[0]['source'][0] == '# Table of Contents\n':
            link_root = _mk_link_root(link_root)
            link_root_for_file = os.path.join(link_root, filename)
            return _mk_link_html(filename, link_root_for_file) \
                   + '\n\n' \
                   + _append_link_root_to_all_pound_hrefs(d[0]['source'][1],
                                                          link_root_for_file)


def ipynb_filepath_list(root_folder='.', recursive=False):
    root_folder = os.path.expanduser(root_folder)
    if recursive:
        return recursive_file_walk_iterator_with_filepath_filter(
            root_folder, filt=lambda x: x.endswith('.ipynb'))
    else:
        return map(lambda x: os.path.abspath(os.path.join(root_folder, x)),
                   filter(lambda x: x.endswith('.ipynb'),
                          os.listdir(root_folder)))


if __name__ == "__main__":
    import argh

    argh.dispatch_command(all_table_of_contents_html_from_notebooks)

    # parser = argh.ArghParser()
    # parser.add_commands([all_table_of_contents_html_from_notebooks,
    #                      table_of_contents_html_from_notebook,
    #                      ipynb_filepath_list])
    # parser.dispatch()
