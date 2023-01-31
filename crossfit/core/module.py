class CrossModule:
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, data, **kwargs):
        return data
