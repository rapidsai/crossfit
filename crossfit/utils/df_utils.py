import functools


def requires_df_backend(*names: str):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from crossfit.dataframe import df_backend

            for name in names:
                try:
                    df_backend(name)
                except ValueError:
                    return False

            return fn(*args, **kwargs)

        return wrapper

    return decorate
