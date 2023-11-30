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

import numpy as np


def create_qrel(relevance_scores, ids=None):
    relevance_scores = np.asarray(relevance_scores, dtype=np.float32)

    # Check for negative relevance scores
    if np.any(relevance_scores < 0):
        raise ValueError("Relevance scores cannot be negative.")

    qrel = {}
    for i, query_scores in enumerate(relevance_scores):
        query_id = ids[i] if ids is not None else f"q{i+1}"
        qrel[query_id] = {}
        for j, score in enumerate(query_scores):
            _score = int(score.item())

            if _score > 0:
                doc_id = f"d{j+1}"
                qrel[query_id][doc_id] = int(score.item())

    return qrel


def create_run(predicted_scores, ids=None):
    predicted_scores = np.asarray(predicted_scores, dtype=np.int32)

    run = {}
    for i, query_scores in enumerate(predicted_scores):
        query_id = ids[i] if ids is not None else f"q{i+1}"
        run[query_id] = {}
        for j, score in enumerate(query_scores):
            doc_id = f"d{j+1}"
            run[query_id][doc_id] = float(score.item())

    return run


def create_results(metric_arrays):
    outputs = {}
    first = next(iter(metric_arrays.values()))

    for i in range(len(first)):
        q_out = {}

        for k, v in metric_arrays.items():
            q_out[k] = float(v[i])

        outputs[f"q{i+1}"] = q_out

    return outputs
