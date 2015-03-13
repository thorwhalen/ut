__author__ = 'thor'

import os
import pattern.web as pweb
from pattern.web import Facebook
from ut.parse.util import disp_html


class FB(Facebook):
    default = dict()
    # default['license'] = os.environ['FB_LICENSE']
    default['license'] = "CAAEuAis8fUgBAI644h9B37fLW47Usi4KzZBUyJSqkhRaUszHCejmsAEJPTGMlF7o3SqLgdP6JcVjMFdj5mPcfdBJjyijudnWaH9KZBV6gE2tEuuqE8KmgHwJZBhe0jZA8rtQQKixO7xZA1n4srzdbWCn7pg70WZAu4L6ZCg8jTpUzwAVirvp51c"

    def __init__(self, **kwargs):
        kwargs = dict(FB.default, **kwargs)
        super(FB, self).__init__(**kwargs)
        # self.key = kwargs['key']
        # # self.fb = pweb_fb

    @staticmethod
    def disp_result_html(res):
        s = ''
        for k in res.keys():
            s += "<b>%s</b>: %s<br>" % (k, res.get(k))
        disp_html(s)