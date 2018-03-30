from __future__ import division


class PredPipeline(object):
    def __init__(self, steps):
        """
        Constructs a callable object that composes the steps listed in the input.
        Important to note that it assigns each step to an attribute (therefore method) of the object, so it's
        different than using normal function composition.
        Originally intended to compose a pipeline of transformers and models with eachother by composing their objects
        (assumed to have a __call__ method).
        :param steps: A list of (func_name, func) pairs defining the pipeline.
        >>> f = PredPipeline(steps=[('f', lambda x: x + 2), ('g', lambda x: x * 10)])
        >>> f(0)
        20
        >>> f(10)
        120
        """
        assert len(steps) > 0, "You need at least one step in your pipeline"
        self.step_names = list()
        for func_name, func in steps:
            assert callable(func), \
                "The object associated with {} wasn't callable".format(func_name)
            setattr(self, func_name, func)
            self.step_names.append(func_name)

    def __call__(self, *args, **kwargs):
        f = getattr(self, self.step_names[0])
        x = f(*args, **kwargs)
        for func_name in self.step_names[1:]:
            f = getattr(self, func_name)
            x = f(x)
        return x
