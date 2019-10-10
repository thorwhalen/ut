__author__ = 'thorwhalen'

import functools

class ParseSearchTerms(object):

    def __init__(self,
                 html_pull=None,
                 html_pull_failure_action=None,
                 parser=None,
                 parser_failure_action=None,
                 parse_diagnosis=None,
                 parse_diagnosis_success_action=None,
                 parse_diagnosis_failure_action=None
                 ):
        self.html_pull = html_pull
        self.html_pull_failure_action = html_pull_failure_action
        self.parser = parser
        self.parser_failure_action = parser_failure_action
        self.parse_diagnosis = parse_diagnosis
        self.parse_diagnosis_failure_action = parse_diagnosis_failure_action
        self.parse_diagnosis_success_action = parse_diagnosis_success_action
        # checking that all attributes are callable
        # for k,v in self.__dict__.items():
        #     assert hasattr(v,'__call__'), "%s is not callable" % k
        # # just extra cool stuff to be more flexible (and examplify some python stuff)
        # attr_dict = self.__dict__
        # for k,v in attr_dict.items():
        #     if v is None:
        #         setattr(self, k, lambda x: universal_mock("mocking %s"%k))
        #     elif not hasattr(v,'__call__'): # if the input property is not callable
        #         # then use it's method "process" (if it has it)
        #         if hasattr(v,'process'):
        #             setattr(self,k,lambda x: getattr(self,k).process(x))
        #         else:
        #             raise AttributeError("ParseSearchTerms attributes must be callable (functions or objects with a __call__ method, or have a process method (named as such)")

    def process(self, search_term):
        try: # getting the html for this search_term
            html = self.html_pull(search_term)
            try: # parsing the html for this search_term
                parsed_result = self.parser(html)
            except RuntimeError as parser_exception:
                self.parser_failure_action(search_term, parser_exception)
            # diagnose the parsed results st_result to decide where to go from here
            diagnosis_is_success, diagnosis_info = self.parse_diagnosis(parsed_result)
            if diagnosis_is_success==True:
                self.parse_diagnosis_success_action(search_term,parsed_result)
            else:
                self.parse_diagnosis_failure_action(search_term,diagnosis_info)
        except RuntimeError as pull_exception:
            self.html_pull_failure_action(search_term, pull_exception)


####### class ParseSearchTerms(object) ends here
#######################################################################################################################


# def universal_mock(message):
#     print message

# below are examples of functions (can be methods of instantiated classes passed as lambda functions) that can be
# passed as properties to ParseSearchTerms()
# I'm showing this so that their interface is clearer

class HtmlPuller(object):
    def __init__(self,html_folder):
        self.html_folder = html_folder
    def get_html(self,t):
        return t + self.x

class parser():
    pass

class Pusher(object):
    def __init__(self,x):
        self.x = x
    def doit(self,t):
        print("%s %d" % (self.x,t))
