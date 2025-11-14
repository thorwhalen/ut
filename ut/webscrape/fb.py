__author__ = 'thor'

import os
import pattern.web as pweb
from pattern.web import Facebook
from ut.parse.util import disp_html


class FB(Facebook):
    default = dict()
    default['license'] = os.environ['FB_LICENSE']
    if default['license'].endswith('\r'):
        default['license'] = default['license'][:-1]

    def __init__(self, **kwargs):
        kwargs = dict(FB.default, **kwargs)
        super().__init__(**kwargs)
        # self.key = kwargs['key']
        # # self.fb = pweb_fb

    @staticmethod
    def disp_result_html(res):
        s = ''
        for k in list(res.keys()):
            s += '<b>{}</b>: {}<br>'.format(k, res.get(k))
        disp_html(s)
