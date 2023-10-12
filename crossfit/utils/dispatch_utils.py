from dask.utils import Dispatch


def supports(dispatch: Dispatch) -> set:
    if dispatch._lazy:
        _to_remove = set()
        for module in dispatch._lazy:
            try:
                register = dispatch._lazy[module]
            except KeyError:
                pass
            else:
                try:
                    register()
                    _to_remove.add(module)
                except ModuleNotFoundError:
                    pass
        for module in _to_remove:
            dispatch._lazy.pop(module, None)

    return set(dispatch._lookup)
