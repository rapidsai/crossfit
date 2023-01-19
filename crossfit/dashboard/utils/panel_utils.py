from functools import wraps

import panel as pn


def parse_html_args(*args):
    wrapped = []
    for arg in args:
        if isinstance(arg, pn.pane.HTML):
            wrapped.append(arg.object)
        else:
            wrapped.append(arg)
    
    return wrapped or args


def html_component(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = parse_html_args(*args)
        return fn(" ".join(args), **kwargs)
    
    return wrapper
