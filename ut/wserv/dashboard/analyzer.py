__author__ = 'thor'


from collections import OrderedDict
from ut.util.pobj import methods_of


class Analyzer:
    """
    An Analyzer is a class to manage a simple dashboard that takes inputs from an html form, and takes action on these
    inputs.

    An Analyzer is defined by a list of dicts, each specifying an form input element in a way that is very similar to
    standard html form elements. Each dict should contain a name and a type. If the type is 'button', the Analyzer
    should have a method of the same name as the 'name' key of the element. This is checked by the
    verify_existence_of_button_functions() method.

    Attributes:
        * input_element_collection: The underlying InputElementCollection
        * button_method: A list of the names of the InputElements that have type='button'
        * input: A dict containing the {key: val} pairs of the forms input values (these are set only when the
        set_inputs(**kwargs) is called.

    The set_input(**kwargs) method updates the input key values (only for those already existing keys)

    The call_button_function(name, **kwargs) updates the input key values (using set_input(**kwargs)) and then calls
    the named method with the **kwargs input.


    """

    def __init__(self, form_elements, to_html_kwargs={}, analyzer_name=''):
        to_html_kwargs = dict(
            prefix='<div id="analyzer_input">\n',
            suffix='<input type="hidden" name="analyzer_name" value="{}" />\n</div>'.format(
                analyzer_name
            ),
            **to_html_kwargs
        )
        self.input_element_collection = InputElementCollection(
            form_elements, to_html_kwargs
        )
        self.button_methods = list()
        self.input = dict()
        for k, v in self.input_element_collection.items():
            if v['type'] == 'button':
                self.button_methods.append(v['name'])
            else:
                self.input.update({k: v.get('value', None)})

    def to_html(self, **kwargs):
        return self.input_element_collection.to_html(**kwargs)

    def verify_existence_of_button_functions(self):
        methods_list = self.method_list()
        for method_name in self.button_methods:
            assert (
                method_name in methods_list
            ), 'This method is missing (is in button input_element): {}'.format(
                method_name
            )

    def method_list(self):
        return methods_of(self)

    def set_inputs(self, **kwargs):
        self.input.update(
            **{k: v for k, v in kwargs.items() if k in list(self.input.keys())}
        )

    def call_button_function(self, name, **kwargs):
        self.set_inputs(**kwargs)
        return self.__getattribute__(name)(**kwargs)

    def __repr__(self):
        s = ''
        for k, v in self.input.items():
            s += f'{k}: {v}\n'
        return s

    def __str__(self):
        return self.to_html()


class InputElementCollection(OrderedDict):
    """
    A InputElementCollection is a class that specifies html form elements.
     It's a collection of InputElements. More precisely, it's an OrderedDict indexed by the name of the InputElements.

    Constructor arguments:
        * form_elements is a list of dicts each specifying an InputElement
        * to_html_kwargs is a dict that specifies how to generate html for a form with the InputElements

    The to_html() method returns an html form with the form elements. The way the html is created can be controlled
    through the constructor's to_html_kwargs argument.
    """

    def __init__(self, form_elements, to_html_kwargs={}):
        self.to_html_kwargs = dict(
            dict(prefix='<form>\n', suffix='</form>\n', sep='\n<br>\n'),
            **to_html_kwargs
        )
        form_element_list = [(x['name'], InputElement(x)) for x in form_elements]
        # for form_element in form_elements:
        #     form_element_list.append((form_element['name'], form_element))
        super().__init__(form_element_list)

    def to_html(self, **kwargs):
        kwargs.update(**self.to_html_kwargs)
        html = kwargs['prefix']
        for k, v in self.items():
            html += v.to_html(**kwargs) + kwargs['sep']
        html += kwargs['suffix']
        return html


class InputElement(dict):
    """
    An InputElement is a dict (but it forces you to have a name and type keys).
    You can enter what ever type you want, but the usage is meant to be aligned with html form attributes
    (see for example http://www.w3schools.com/tags/att_input_type.asp).

    There are other keys that are not standard attributes:
        * the display key specifies what text will be displayed before the input box. If the display key is not
        specified, it will be created and given the value of name. Therefore, if you want to display nothing, you must
        specify display="". The type='button' input elements are treated slightly differently. If display key is not
        specified, it will not be created. But if the value key is not specified, it will take on the value of the
        name attribute.

    The to_html() method returns a string that corresponds to the html for the input element.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verify_inputs()

    def verify_inputs(self):
        key_list = list(self.keys())
        assert 'name' in key_list, "you need to have a 'name' in a InputElement"
        assert (
            'type' in key_list
        ), "you need to have a 'type' in a InputElement (see html input tag types)"
        if self['type'] == 'button':
            if not self.get('value'):
                self['value'] = self['name']
        else:
            if not self.get('display'):
                self['display'] = self['name']
        # if self['type'] == 'button':
        #     if 'function' not in key_list:
        #         self.update(function=self['name'])

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.verify_inputs()

    def to_html(self, **kwargs):
        d = self.copy()
        element_type = d['type']
        element_name = d['name']
        if element_type == 'input':
            html = ''
            d['value'] == d.pop('display', '')
        else:
            html = d.pop('display', '')
            if html == ': ':
                html = ''
        html += '<input type="{type}" name="{name}"'.format(
            type=d.pop('type'), name=d.pop('name')
        )
        # d.pop('function', None)  # get rid of function
        for k, v in d.items():
            html += f' {k}="{v}"'
        if element_type == 'button':
            html += f' onclick="getResult(\'{element_name}\')"'
        html += '>'
        return html
