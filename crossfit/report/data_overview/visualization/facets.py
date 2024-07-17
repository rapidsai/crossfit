# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os

from tensorflow_metadata.proto.v0 import statistics_pb2

STATS_FILE_NAME = "stats.pb"


def _maybe_to_pandas(data):
    # Utility to convert cudf data to pandas (for now)
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return data


def visualize(con_df=None, cat_df=None, name="data") -> "FacetsOverview":
    output = statistics_pb2.DatasetFeatureStatisticsList()
    datasets = {}

    if con_df is not None:
        if hasattr(con_df.index, "name") and con_df.index.name == "column":
            con_df.columns = con_df.columns.str.replace("ContinuousMetrics.", "")

            datasets[name] = statistics_pb2.DatasetFeatureStatistics(
                name=name, num_examples=int(con_df["common_stats.count"].iloc[0])
            )

            for row_name, row in con_df.iterrows():
                feature = datasets[name].features.add()
                feature.name = row_name
                feature.num_stats.CopyFrom(_create_numeric_statistics(row))
        else:
            for row_name, row in con_df.iterrows():
                row.index = row.index.str.replace("ContinuousMetrics.", "")
                data_name = f"{row_name[0]}={row_name[1]}"

                datasets[data_name] = statistics_pb2.DatasetFeatureStatistics(
                    name=data_name, num_examples=int(row["common_stats.count"])
                )

                feature = datasets[data_name].features.add()
                feature.name = row_name[-1]
                feature.num_stats.CopyFrom(_create_numeric_statistics(row))

    if cat_df is not None:
        if hasattr(cat_df.index, "name") and cat_df.index.name == "column":
            cat_df.columns = cat_df.columns.str.replace("CategoricalMetrics.", "")

            if name not in datasets:
                datasets[name] = statistics_pb2.DatasetFeatureStatistics(
                    name=name, num_examples=int(cat_df["common_stats.count"].iloc[0])
                )

            for row_name, row in cat_df.iterrows():
                feature = datasets[name].features.add()
                feature.name = row_name
                feature.string_stats.CopyFrom(_create_string_statistics(row))
        else:
            for row_name, row in cat_df.iterrows():
                row.index = row.index.str.replace("CategoricalMetrics.", "")
                data_name = f"{row_name[0]}={row_name[1]}"

                if data_name not in datasets:
                    datasets[data_name] = statistics_pb2.DatasetFeatureStatistics(
                        name=data_name, num_examples=int(row["common_stats.count"])
                    )

                feature = datasets[data_name].features.add()
                feature.name = row_name[-1]
                feature.string_stats.CopyFrom(_create_string_statistics(row))

    for dataset in datasets.values():
        d = output.datasets.add()
        d.CopyFrom(dataset)

    return FacetsOverview(output)


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
        num_non_missing=int(row["common_stats.count"]),
        num_missing=int(row["common_stats.num_missing"]),
        min_num_values=1,
        max_num_values=1,
        avg_num_values=1,
    )


def _create_numeric_statistics(row) -> statistics_pb2.NumericStatistics:
    common_stats = _create_common_stats(row)
    return statistics_pb2.NumericStatistics(
        min=row["range.min"],
        max=row["range.max"],
        # histograms=[hist],    # TODO: Add histogram
        common_stats=common_stats,
        # TODO: Add median
        mean=row["moments.mean"],
        std_dev=row["moments.std"],
        # num_zeros=dask_stats[col]["num_zeroes"].item(),
    )


def _create_string_statistics(row) -> statistics_pb2.StringStatistics:
    common_stats = _create_common_stats(row)
    string_stats = statistics_pb2.StringStatistics(
        common_stats=common_stats,
        unique=row["value_counts.num_unique"],
        avg_length=row["mean_str_len"],
    )

    top_k = zip(
        list(row["value_counts.top_values"]),
        list(row["value_counts.top_counts"]),
    )
    ranks = string_stats.rank_histogram
    for k, (val, freq) in enumerate(top_k):
        f = string_stats.top_values.add()
        f.value = val
        f.frequency = freq
        b = ranks.buckets.add()
        b.CopyFrom(
            statistics_pb2.RankHistogram.Bucket(
                low_rank=k, high_rank=k, label=val, sample_count=freq
            )
        )

    return string_stats
