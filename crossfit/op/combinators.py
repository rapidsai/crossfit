from crossfit.op.base import Op


class Sequential(Op):
    def __init__(self, *ops, pre=None, cols=False, repartition=None, keep_cols=None):
        super().__init__(pre=pre, cols=cols, keep_cols=keep_cols)
        self.ops = ops
        self.repartition = repartition

        if keep_cols:
            for op in self.ops:
                op.keep_cols.extend(keep_cols)

    def call_dask(self, data):
        for op in self.ops:
            if self.repartition is not None:
                data = data.repartition(self.repartition)

            data = op(data)

        return data

    def call(self, data):
        for op in self.ops:
            data = op(data)

        return data
