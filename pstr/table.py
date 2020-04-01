"""
Utils that related tables to strings.
"""
import csv
from io import StringIO


def linesstr_to_listrows(linesstr, ncols, line_sep='\n'):
    """

    Args:
        linesstr: a string
        ncols: number of columns -- equivalently, the size of a group of lines
        line_sep: The string that marks the separation of one line and another

    Returns:

    >>> linestr = 'A\\nB\\nC\\n1\\n2\\n3'
    >>> list(linesstr_to_listrows(linestr, 3))
    [('A', 'B', 'C'), ('1', '2', '3')]
    """
    return zip(*[iter(linesstr.split(line_sep))] * ncols)


def listrows_to_csvstr(listrows, dialect='excel', **fmtparams):
    """

    >>> listrows_to_csvstr([('A', 'B', 'C'), ('1', '2', '3')])
    'A,B,C\\r\\n1,2,3\\r\\n'

    """
    t = StringIO()
    w = csv.writer(t, dialect=dialect, **fmtparams)
    w.writerows(listrows)
    t.seek(0)
    s = t.read()
    t.close()
    return s


def linesstr_to_csvstr(linesstr, ncols, line_sep='\n', dialect='excel', **fmtparams):
    listrows = linesstr_to_listrows(linesstr, ncols, line_sep=line_sep)
    return listrows_to_csvstr(listrows, dialect, **fmtparams)


def rowstr_to_md(row):
    return '| ' + ' | '.join(row) + ' |'


def listrows_to_md(listrows, center_aligned_columns=None, right_aligned_columns=None):
    """
    >>> print(listrows_to_md([('A', 'B', 'C'), ('1', '2', '3')]))
    | A | B | C |
    | - | - | - |
    | 1 | 2 | 3 |
    """
    widths = list(map(max, zip(*[list(map(len, row)) for row in listrows])))
    rows = [rowstr_to_md([cell.ljust(width) for cell, width in zip(row, widths)]) for row in listrows]
    separators = ['-' * width for width in widths]

    if right_aligned_columns is not None:
        for column in right_aligned_columns:
            separators[column] = ('-' * (widths[column] - 1)) + ':'
    if center_aligned_columns is not None:
        for column in center_aligned_columns:
            separators[column] = ':' + ('-' * (widths[column] - 2)) + ':'

    rows.insert(1, rowstr_to_md(separators))

    return '\n'.join(rows)


def linesstr_to_md(linesstr, ncols, line_sep='\n',
                   center_aligned_columns=None, right_aligned_columns=None):
    """
    >>> print(linesstr_to_md('A\\nB\\nC\\n1\\n2\\n3'))
    | A | B | C |
    | - | - | - |
    | 1 | 2 | 3 |
    """
    listrows = list(linesstr_to_listrows(linesstr, ncols, line_sep=line_sep))
    return listrows_to_md(listrows, center_aligned_columns, right_aligned_columns)
