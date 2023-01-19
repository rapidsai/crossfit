from functools import wraps

import panel as pn


def parse_html_args(*args):
    wrapped = []
    for arg in args:
        if isinstance(arg, pn.pane.HTML) or hasattr(arg, "object"):
            wrapped.append(arg.object)
        else:
            wrapped.append(arg)
    
    return wrapped or args


def html_component(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = parse_html_args(*args)
        if args:
            args = " ".join(args) if len(args) > 1 else args[0]
        return fn(args, **kwargs)
    
    return wrapper
