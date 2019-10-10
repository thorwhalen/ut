

from functools import partial


def get_step_func_name_obj_and_call_method(step):
    if len(step) == 2:
        return step[0], step[1], '__call__'
    elif len(step) == 3:
        return step[0], step[1], step[2]
    else:
        return step[0], step[1], step[2:]


class ComputationPipeline(object):
    def __init__(self, steps):
        """
        Constructs a callable object that composes the steps listed in the input.
        Important to note that it assigns each step to an attribute (therefore method) of the object, so it's
        different than using normal function composition.
        Originally intended to compose a pipeline of transformers and models with eachother by composing their objects
        (assumed to have a __call__ method).
        :param steps: A list of (func_name, func) pairs defining the pipeline.
        >>> f = ComputationPipeline(steps=[('f', lambda x: x + 2), ('g', lambda x: x * 10)])
        >>> f(0)
        20
        >>> f(10)
        120
        """
        assert len(steps) > 0, "You need at least one step in your pipeline"
        self.step_names = list()
        self.call_method_for_func_name = dict()
        for func_name, obj, call_method in map(get_step_func_name_obj_and_call_method, steps):
            setattr(self, func_name, obj)  # make an attribute with the func_name, pointing to object
            self.step_names.append(func_name)
            func = getattr(obj, call_method)
            if isinstance(call_method, str):
                self.call_method_for_func_name[func_name] = func
            else:
                if len(call_method) == 2 and isinstance(call_method[0], str):
                    partial_args_and_keywords = call_method[1]
                    call_method = call_method[0]
                    if isinstance(partial_args_and_keywords, dict):
                        self.call_method_for_func_name[func_name] = partial(func,
                                                                            *partial_args_and_keywords['args'],
                                                                            **partial_args_and_keywords['keywords'])
                    else:
                        raise TypeError("Don't know how to handle that type of call_method spec.")
                else:
                    raise TypeError("Don't know how to handle that type of call_method spec.")

    def __call__(self, *args, **kwargs):
        x = self.call_method_for_func_name[self.step_names[0]](*args, **kwargs)
        for func_name in self.step_names[1:]:
            x = self.call_method_for_func_name[func_name](x)
        return x
