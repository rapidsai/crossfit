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

import os
import shutil

import dask.dataframe as dd

from crossfit.dataset.base import IRDataset
from crossfit.dataset.beir.raw import download_raw, sample_raw
from crossfit.dataset.home import CF_HOME


def load_dataset(
    name,
    out_dir=None,
    blocksize=2**30,
    overwrite=False,
) -> IRDataset:
    raw_path = download_raw(name, out_dir=out_dir, overwrite=False)

    return _process_data(name, raw_path, blocksize=blocksize, overwrite=overwrite, out_dir=out_dir)


def load_test_dataset(
    name,
    out_dir=None,
    blocksize=2**30,
    overwrite=False,
) -> IRDataset:
    raw_path = sample_raw(name, out_dir=out_dir, overwrite=False)

    return _process_data(
        name, raw_path, blocksize=blocksize, overwrite=overwrite, out_dir=out_dir, is_test=True
    )


def _process_data(name, raw_path, blocksize=2**30, overwrite=False, out_dir=None, is_test=False):
    import dask_cudf

    out_dir = out_dir or CF_HOME
    processed_name = "processed-test" if is_test else "processed"
    processed_dir = os.path.join(out_dir, processed_name, name)

    # Check if the output directory already exists
    if os.path.exists(processed_dir):
        if overwrite:
            print("Processed directory {} already exists. Overwriting.".format(processed_dir))
            shutil.rmtree(processed_dir)  # Remove the existing directory
        else:
            print(
                "Processed directory {} already exists. Skipping processing.".format(processed_dir)
            )

            return IRDataset.from_dir(processed_dir)

    os.makedirs(processed_dir, exist_ok=True)

    print("Converting queries...")
    queries_dir = os.path.join(processed_dir, "queries")
    queries_ddf = dask_cudf.read_json(
        os.path.join(raw_path, "queries.jsonl"),
        lines=True,
        blocksize=blocksize,
        dtype={"_id": "string", "text": "string"},
    )[["_id", "text"]]
    queries_ddf = reset_global_index(queries_ddf)
    queries_ddf.to_parquet(queries_dir)

    print("Converting corpus...")
    corpus_dir = os.path.join(processed_dir, "corpus")
    corpus_ddf = dask_cudf.read_json(
        os.path.join(raw_path, "corpus.jsonl"),
        lines=True,
        blocksize=blocksize,
        dtype={"_id": "string", "title": "string", "text": "string"},
    )[["_id", "title", "text"]]
    corpus_ddf = reset_global_index(corpus_ddf)
    corpus_ddf.to_parquet(corpus_dir)

    qrels_dir = os.path.join(processed_dir, "qrels")
    qrels_files = [f for f in os.listdir(os.path.join(raw_path, "qrels")) if f.endswith(".tsv")]
    qrels_dtypes = {"query-id": "str", "corpus-id": "str", "score": "int32"}
    dataset_dirs = {"query": queries_dir, "item": corpus_dir}
    name_mapping = {"train": "train", "dev": "val", "test": "test"}
    for qrels_file in qrels_files:
        print(f"Processing {qrels_file}...")
        qrels_ddf = dask_cudf.read_csv(
            os.path.join(raw_path, "qrels", qrels_file),
            sep="\t",
            dtype=qrels_dtypes,
        )[["query-id", "corpus-id", "score"]]
        qrels_ddf = (
            qrels_ddf.merge(
                queries_ddf,
                left_on="query-id",
                right_on="_id",
                how="left",
            )
            .rename(columns={"text": "query", "index": "query-index"})
            .merge(
                corpus_ddf,
                left_on="corpus-id",
                right_on="_id",
                how="left",
            )
            .rename(columns={"index": "corpus-index"})
        )[
            [
                "query-id",
                "corpus-id",
                "query-index",
                "corpus-index",
                "title",
                "query",
                "text",
                "score",
            ]
        ]
        qrels_path = os.path.join(qrels_dir, qrels_file.replace(".tsv", ""))
        dataset_dirs[name_mapping[qrels_file.split(".")[0]]] = qrels_path
        qrels_ddf.to_parquet(qrels_path)

    return IRDataset.from_dir(processed_dir)


def reset_global_index(ddf: dd.DataFrame, index_col: str = "index") -> dd.DataFrame:
    ddf[index_col] = 1
    ddf[index_col] = ddf[index_col].cumsum() - 1

    return ddf.set_index(index_col, sorted=True, drop=False)
