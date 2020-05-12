import re

assert_re = re.compile('^assert\ ')
# assertion_re = re.compile('^assert\ |\ ==\ .+$')
assertion_capture_re = re.compile('(^>>>\ assert\ )(?P<statement>.+)(\ ==\ )(?P<val>.+$)')
assertion_as_doctest_templ = ">>> {statement}\n{val}\n"
space_re = re.compile('\s')


def remove_assertion(txt):
    lines = txt.split('\n')
    f = lambda line: re.compile('^assert\ |\ ==\ .+$').sub('', line)
    return '\n'.join(map(f, lines))


def assertions_to_doctest(txt):
    lines = txt.split('\n')
    new_txt = ''
    for line in lines:
        m = assertion_capture_re.match(line)
        # print(line, m)
        if m:
            new_txt += assertion_as_doctest_templ.format(**m.groupdict())
        else:
            new_txt += line + '\n'
    return new_txt


def _txt_to_doctest_txt_for_line(line):
    if len(line) == 0:
        return '>>> '
    elif space_re.match(line[0]):
        return '...\t' + line
    else:
        return '>>> ' + line


def txt_to_doctest_txt(txt, assertions_as_repl=True):
    return '\n'.join(map(_txt_to_doctest_txt_for_line, txt.split('\n')))
