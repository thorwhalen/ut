class SimpleProperty:
    def __get__(self, instance, owner):
        return instance


class DelegatedAttribute:
    def __init__(self, delegate_name, attr_name):
        self.attr_name = attr_name
        self.delegate_name = delegate_name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return getattr(self.delegate(instance), self.attr_name)

    def __set__(self, instance, value):
        # instance.delegate.attr = value
        setattr(self.delegate(instance), self.attr_name, value)

    def __delete__(self, instance):
        delattr(self.delegate(instance), self.attr_name)

    def delegate(self, instance):
        return getattr(instance, self.delegate_name)


def delegate_as(delegate_cls, to='delegate', include=frozenset(), ignore=frozenset()):
    # turn include and ignore into sets, if they aren't already
    if not isinstance(include, set):
        include = set(include)
    if not isinstance(ignore, set):
        ignore = set(ignore)
    delegate_attrs = set(delegate_cls.__dict__.keys())
    attributes = include | delegate_attrs - ignore

    def inner(cls):
        # create property for storing the delegate
        setattr(cls, to, SimpleProperty())
        # don't bother adding attributes that the class already has
        attrs = attributes - set(cls.__dict__.keys())
        # set all the attributes
        for attr in attrs:
            setattr(cls, attr, DelegatedAttribute(to, attr))
        return cls

    return inner


class DelegateTo:  # Read only alt to DelegatedAttribute
    def __init__(self, to, method=None):
        if to == 'self' and method is None:
            raise ValueError("DelegateTo('self') is invalid, provide 'method' too")
        self.to = to
        self.method = method

    def __get__(self, instance, owner):
        if self.to == 'self':
            return getattr(instance, self.method)
        if self.method is not None:
            return getattr(getattr(instance, self.to), self.method)
        for method, v in instance.__class__.__dict__.items():
            if v is self:
                self.method = method
                return getattr(getattr(instance, self.to), method)
