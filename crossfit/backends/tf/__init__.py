from crossfit.backends.tf.array import *  # noqa: F401, F403

try:
    import tensorflow as tf

    del tf

    from crossfit.backends.tf.metrics import (  # noqa: F401, F403
        from_tf_metric,
        to_tf_metric,
    )

except ImportError:
    pass
