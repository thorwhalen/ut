from __future__ import division

from ut.pdict.diagnosis import validate_kwargs, base_validation_funs

dflt_validation_funs = base_validation_funs
dflt_all_kwargs_should_be_in_validation_dict = False
dflt_ignore_misunderstood_validation_instructions = False

dflt_arg_pattern = r'.+'


class NamingInterface(object):
    def __init__(self,
                 params=None,
                 validation_funs=dflt_validation_funs,
                 all_kwargs_should_be_in_validation_dict=dflt_all_kwargs_should_be_in_validation_dict,
                 ignore_misunderstood_validation_instructions=dflt_ignore_misunderstood_validation_instructions,
                 **kwargs):
        if params is None:
            params = {}

        validation_dict = {var: info.get('validation', {}) for var, info in params.iteritems()}
        default_dict = {var: info.get('default', None) for var, info in params.iteritems()}
        arg_pattern = {var: info.get('arg_pattern', dflt_arg_pattern) for var, info in params.iteritems()}
        named_arg_pattern = {var: r'(?P<buf_size_frm>' + pat + ')' for var, pat in arg_pattern.iteritems()}
        to_str = {var: info['to_str'] for var, info in params.iteritems() if 'to_str' in info}
        to_val = {var: info['to_val'] for var, info in params.iteritems() if 'to_val' in info}

        self.validation_dict = validation_dict
        self.default_dict = default_dict
        self.arg_pattern = arg_pattern
        self.named_arg_pattern = named_arg_pattern
        self.to_str = to_str
        self.to_val = to_val

        self.validation_funs = validation_funs
        self.all_kwargs_should_be_in_validation_dict = all_kwargs_should_be_in_validation_dict
        self.ignore_misunderstood_validation_instructions = ignore_misunderstood_validation_instructions

    def validate_kwargs(self, **kwargs):
        return validate_kwargs(kwargs_to_validate=kwargs,
                               validation_dict=self.validation_dict,
                               validation_funs=self.validation_funs,
                               all_kwargs_should_be_in_validation_dict=self.all_kwargs_should_be_in_validation_dict,
                               ignore_misunderstood_validation_instructions=self.ignore_misunderstood_validation_instructions)

    def default_for(self, arg, **kwargs):
        default = self.default_dict[arg]
        if not isinstance(default, dict) or 'args' not in default or 'func' not in default:
            return default
        else:  # call the func on the default['args'] values given in kwargs
            args = {arg_: kwargs[arg_] for arg_ in default['args']}
            return default['func'](*args)

    def str_kwargs_from(self, **kwargs):
        return {k: self.to_str[k](v) for k, v in kwargs.iteritems() if k in self.to_str}

    def val_kwargs_from(self, **kwargs):
        return {k: self.to_val[k](v) for k, v in kwargs.iteritems() if k in self.to_val}

    def name_for(self, **kwargs):
        raise NotImplementedError("Interface method: Method needs to be implemented")

    def info_for(self, **kwargs):
        raise NotImplementedError("Interface method: Method needs to be implemented")

    def is_valid_name(self, name):
        raise NotImplementedError("Interface method: Method needs to be implemented")


class Naming(NamingInterface):
    def __init__(self, validation_dict=None, **kwargs):
        super(Naming, self).__init__(validation_dict=validation_dict, **kwargs)
