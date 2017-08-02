from __future__ import division


def to_curl(request, headers="Content-Type: application/json"):
    """
    Get a curl string from a python request.
    :param request: a requests.request object
    :return: a curl string corresponding to this request
    """

    if hasattr(request, 'request'):
        request = getattr(request, 'request')

    if headers is None:
        headers = {k: v for k, v in request.headers.iteritems()}

    # headers = headers or {k: v for k, v in request.headers.iteritems()}

    if isinstance(headers, dict):
        headers = ["'{}: {}'".format(k, v) for k, v in headers.iteritems()]
        headers = " -H ".join(sorted(headers))
    elif isinstance(headers, list):
        headers = " -H ".join(map(lambda x: '"' + x + '"', sorted(headers)))
    elif isinstance(headers, basestring) and not headers.startswith("'") or not headers.startswith('"'):
        headers = '"' + headers + '"'

    command = "curl -X {method} -H {headers} -d '{data}' '{uri}'".format(
        data=request.body or "",
        headers=headers,
        method=request.method,
        uri=request.url,
    )
    return command
