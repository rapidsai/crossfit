import abc


class Report(abc.ABC):
    def visualize(self, *args, **kwargs):
        raise NotImplementedError()

    def compare(self, other):
        raise NotImplementedError()
