import crossfit as cf


def test_pre():
    def pre(x):
        return x + 1

    class Test(cf.Aggregator):
        def prepare(self, data):
            return {"test": data + 1}

    agg = cf.Aggregator({"inc": pre}, pre=pre)
    agg_extended = Test({"inc": pre}, pre=pre)

    assert agg.prepare(0)["inc"] == 2
    assert agg_extended.prepare(0)["test"] == 2
