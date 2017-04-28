from __future__ import division

import os, json
# import IPython.core.display as idisp
# from ut.pfile.name import recursive_file_walk_iterator
from ut.pstr.to import file as str_to_file

notebook_root = 'http://localhost:8888/notebooks/'
default_notebook_folder = 'soto'
default_link_root = os.path.join(notebook_root, default_notebook_folder)


def all_table_of_contents_html_from_notebooks_in_folder(folder='.', save_to_file=None):
    folder = os.path.abspath(folder)
    folder_name = os.path.dirname(folder)
    s = "<b>{}</b><br><br>\n\n".format(folder)
    for f in ipynb_filepath_list(folder):
        #         print f
        ss = table_of_contents_html_from_notebook(f)
        if ss is not None:
            s += ss + '<br>\n\n'
    if save_to_file is None:
        return s
    else:
        str_to_file(s, save_to_file)


def _mk_link_html(filename, link_root=default_link_root):
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
            link_root_for_file = os.path.join(link_root, filename)
            return _mk_link_html(filename, link_root_for_file) \
                   + '\n\n' \
                   + _append_link_root_to_all_pound_hrefs(d[0]['source'][1],
                                                          link_root_for_file)


def ipynb_filepath_list(root_folder='.'):
    return map(lambda x: os.path.abspath(os.path.join(root_folder, x)),
               filter(lambda x: x.endswith('.ipynb'),
                      os.listdir(root_folder)))
