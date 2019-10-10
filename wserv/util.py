


def to_curl(request, headers="simple", print_it=True):
    """
    Get a curl string from a python request.
    :param request: a requests.models.Response or a requests.models.Response.request object
    :param headers: headers to include in the curl request (the "-H" args).
        Specified by
            a {header_name: header_value} dict
            a "header_name: header_value" string (if only one header}
            a list of "header_name: header_value" strings
        If None, will ask the request objects for it's request headers.
    :param print_it: If True (default), will print the curl command. If not, it will return a string containing it.

    :return: a curl string corresponding to this request
    """

    if hasattr(request, 'request'):
        request = getattr(request, 'request')

    command = None

    if headers is None:
        headers = {k: v for k, v in request.headers.items()}
    elif isinstance(headers, str) and headers == 'simple':
        if request.method == 'GET':
            command = "curl -X {method} '{uri}'".format(
                        method=request.method,
                        uri=request.url,
                    )
        elif request.method == 'POST':
            command = "curl -X {method} -H 'Content-Type: application/json' -d '{data}' '{uri}'".format(
                data=request.body or "",
                method=request.method,
                uri=request.url,
            )

    if command is None:
        if isinstance(headers, dict):
            headers = ["'{}: {}'".format(k, v) for k, v in headers.items()]
            headers = " -H ".join(sorted(headers))
        elif isinstance(headers, list):
            headers = " -H ".join(['"' + x + '"' for x in sorted(headers)])
        elif isinstance(headers, str) and (not headers.startswith("'") or not headers.startswith('"')):
                headers = '"' + headers + '"'

        command = "curl -X {method} -H {headers} -d '{data}' '{uri}'".format(
            data=request.body or "",
            headers=headers,
            method=request.method,
            uri=request.url,
        )

    if print_it:
        print(command)
    else:
        return command
