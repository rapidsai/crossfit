from crossfit.op.base import Op


class Sequential(Op):
    def __init__(self, *ops, pre=None, cols=False, repartition=None):
        super().__init__(pre=pre, cols=cols)
        self.ops = ops
        self.repartition = repartition

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
