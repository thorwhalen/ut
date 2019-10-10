


class ClientError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


class EmptyResponse(ClientError):
    status_code = 400

    def __init__(self, message, payload=None):
        super(EmptyResponse, self).__init__(message, self.status_code, payload)


class BadRequest(ClientError):
    status_code = 400

    def __init__(self, message, payload=None):
        super(BadRequest, self).__init__(message, self.status_code, payload)


class Forbidden(ClientError):
    status_code = 403

    def __init__(self, message, payload=None):
        super(Forbidden, self).__init__(message, self.status_code, payload)


class NotFound(ClientError):
    status_code = 404

    def __init__(self, message, payload=None):
        super(NotFound, self).__init__(message, self.status_code, payload)


class MethodNotAllowed(ClientError):
    status_code = 405

    def __init__(self, message, payload=None):
        super(MethodNotAllowed, self).__init__(message, self.status_code, payload)


class NotAcceptable(ClientError):
    status_code = 406

    def __init__(self, message, payload=None):
        super(NotAcceptable, self).__init__(message, self.status_code, payload)


class RequestTimeOut(ClientError):
    status_code = 408

    def __init__(self, message, payload=None):
        super(RequestTimeOut, self).__init__(message, self.status_code, payload)


class Conflict(ClientError):
    status_code = 409

    def __init__(self, message, payload=None):
        super(Conflict, self).__init__(message, self.status_code, payload)


class Gone(ClientError):
    status_code = 410

    def __init__(self, message, payload=None):
        super(Gone, self).__init__(message, self.status_code, payload)


class LengthRequired(ClientError):
    status_code = 411

    def __init__(self, message, payload=None):
        super(LengthRequired, self).__init__(message, self.status_code, payload)


class PreconditionFailed(ClientError):
    status_code = 412

    def __init__(self, message, payload=None):
        super(PreconditionFailed, self).__init__(message, self.status_code, payload)


class PayloadTooLarge(ClientError):
    status_code = 413

    def __init__(self, message, payload=None):
        super(PayloadTooLarge, self).__init__(message, self.status_code, payload)


class ExpectationFailed(ClientError):
    status_code = 417

    def __init__(self, message, payload=None):
        super(ExpectationFailed, self).__init__(message, self.status_code, payload)


class ForbiddenAttribute(Forbidden):
    def __init__(self, attr, payload=None):
        super(self.__class__, self).__init__("Forbidden attribute: " + attr, payload)


class ForbiddenMethod(Forbidden):
    def __init__(self, method, payload=None):
        super(self.__class__, self).__init__("Forbidden method: " + method, payload)


class ForbiddenProperty(Forbidden):
    def __init__(self, property, payload=None):
        super(self.__class__, self).__init__("Forbidden property: " + property, payload)


class MissingAttribute(BadRequest):
    def __init__(self, message="No attribute (method or property) was specified.", payload=None):
        super(self.__class__, self).__init__(message, payload)


class MissingMethod(BadRequest):
    def __init__(self, message="No method was specified.", payload=None):
        super(self.__class__, self).__init__(message, payload)


class MissingProp(BadRequest):
    def __init__(self, payload=None):
        super(self.__class__, self).__init__("No prop was specified.", payload)


class UnknownMethod(BadRequest):
    def __init__(self, method, payload=None):
        super(self.__class__, self).__init__("Unknown method: {}".format(method), payload)


class UnknownProp(BadRequest):
    def __init__(self, prop, payload=None):
        super(self.__class__, self).__init__("Unknown prop: {}".format(prop), payload)


class UnknownParameters(BadRequest):
    def __init__(self, message, payload=None):
        super(self.__class__, self).__init__(message, payload)


class UnknownError():
    pass
