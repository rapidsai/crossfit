import base64
import os

from tensorflow_metadata.proto.v0 import statistics_pb2

from crossfit.core.frame import MetricFrame


STATS_FILE_NAME = "stats.pb"


def visualize(
    con_mf: MetricFrame, cat_mf: MetricFrame, name="data"
) -> "FacetsOverview":
    data = statistics_pb2.DatasetFeatureStatisticsList()

    if con_mf.data is None:
        stats = None
        if con_mf is not None:
            con_result = con_mf.result()

            if stats is None:
                stats = statistics_pb2.DatasetFeatureStatistics(
                    name=name, num_examples=int(con_result.iloc[0]["common.count"])
                )

            for row_name, row in con_result.iterrows():
                feature = stats.features.add()
                feature.name = row_name
                common_stats = _create_common_stats(row)
                feature.num_stats.CopyFrom(
                    statistics_pb2.NumericStatistics(
                        min=row["range.min"],
                        max=row["range.max"],
                        # histograms=[hist],    # TODO: Add histogram
                        common_stats=common_stats,
                        # TODO: Add median
                        mean=row["moments.mean"],
                        std_dev=row["moments.std"],
                        # num_zeros=dask_stats[col]["num_zeroes"].item(),
                    )
                )

        if cat_mf is not None:
            cat_result = cat_mf.result()

            if stats is None:
                stats = statistics_pb2.DatasetFeatureStatistics(
                    name=name, num_examples=int(cat_result.iloc[0]["common.count"])
                )

            for row_name, row in cat_result.iterrows():
                feature = stats.features.add()
                feature.name = row_name
                common_stats = _create_common_stats(row)
                feature.string_stats.CopyFrom(
                    statistics_pb2.StringStatistics(
                        common_stats=common_stats,
                        unique=row["num_unique"],
                        avg_length=row["str_len"],
                    )
                )

                ranks = feature.string_stats.rank_histogram
                for k, (val, freq) in enumerate(
                    zip(row["top_values"], row["top_counts"])
                ):
                    f = feature.string_stats.top_values.add()
                    f.value = val
                    f.frequency = freq
                    b = ranks.buckets.add()
                    b.CopyFrom(
                        statistics_pb2.RankHistogram.Bucket(
                            low_rank=k, high_rank=k, label=val, sample_count=freq
                        )
                    )

        d = data.datasets.add()
        d.CopyFrom(stats)
    else:
        raise ValueError("Group-by metric-frame is not supported yet.")

    return FacetsOverview(data)


def load(path, file_name=STATS_FILE_NAME) -> "FacetsOverview":
    return FacetsOverview.load(path, file_name)


class FacetsOverview:
    HTML_TEMPLATE = """
<script
 src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
<link rel="import"
 href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
<facets-overview id="elem"></facets-overview>
<script>
 document.querySelector("#elem").protoInput = "{protostr}";
</script>"""

    def __init__(self, stats: statistics_pb2.DatasetFeatureStatisticsList):
        self.stats = stats

    def display_overview(self):
        from IPython.core.display import HTML, display

        return display(HTML(self.to_html()))

    def to_html(self):
        protostr = self.to_proto_string(self.stats)
        html = self.HTML_TEMPLATE.format(protostr=protostr)

        return html

    def save_to_html(self, output_dir, file_name="stats.html"):
        with open(os.path.join(output_dir, file_name), "w") as html_file:
            html_file.write(self.to_html())

    def to_proto_string(self, inputs):
        return base64.b64encode(inputs.SerializeToString()).decode("utf-8")

    def save(self, output_dir, file_name=STATS_FILE_NAME):
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, "wb") as f:
            f.write(self.stats.SerializeToString())

        self.save_to_html(output_dir)

    @classmethod
    def load(cls, path, file_name=STATS_FILE_NAME) -> "FacetsOverview":
        if path.endswith(".pb"):
            stats_file = path
        else:
            stats_file = os.path.join(path, file_name)
        stats = statistics_pb2.DatasetFeatureStatisticsList()
        with open(stats_file, "rb") as f:
            stats.ParseFromString(f.read())

        return cls(stats)

    def _repr_html_(self):
        # Repr for Jupyter Notebook
        return self.to_html()


def _create_common_stats(row) -> statistics_pb2.CommonStatistics:
    return statistics_pb2.CommonStatistics(
        num_non_missing=int(row["common.count"]),
        num_missing=int(row["common.num_missing"]),
        min_num_values=1,
        max_num_values=1,
        avg_num_values=1,
    )
