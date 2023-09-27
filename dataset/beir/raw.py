import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Union

from beir import util
from crossfit.dataset.home import CF_HOME


@dataclass
class DatasetInfo:
    dataset: str
    website: str
    is_public: bool
    dataset_type: List[str]
    queries: int
    corpus_size: str
    rel_dq: float
    download_link: Union[str, None]
    md5: Union[str, None]


BEIR_DATASETS: Dict[str, DatasetInfo] = {
    "msmarco": DatasetInfo(
        "MSMARCO",
        "https://microsoft.github.io/msmarco/",
        True,
        ["train", "dev", "test"],
        6980,
        "8.84M",
        1.1,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip",
        "444067daf65d982533ea17ebd59501e4",
    ),
    "trec-covid": DatasetInfo(
        "TREC-COVID",
        "https://ir.nist.gov/covidSubmit/index.html",
        True,
        ["test"],
        50,
        "171K",
        493.5,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
        "ce62140cb23feb9becf6270d0d1fe6d1",
    ),
    "nfcorpus": DatasetInfo(
        "NFCorpus",
        "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        True,
        ["train", "dev", "test"],
        323,
        "3.6K",
        38.2,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "a89dba18a62ef92f7d323ec890a0d38d",
    ),
    "nq": DatasetInfo(
        "NQ",
        "https://ai.google.com/research/NaturalQuestions",
        True,
        ["train", "test"],
        3452,
        "2.68M",
        1.2,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
        "d4d3d2e48787a744b6f6e691ff534307",
    ),
    "hotpotqa": DatasetInfo(
        "HotpotQA",
        "https://hotpotqa.github.io",
        True,
        ["train", "dev", "test"],
        7405,
        "5.23M",
        2.0,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
        "f412724f78b0d91183a0e86805e16114",
    ),
    "fiqa": DatasetInfo(
        "FiQA-2018",
        "https://sites.google.com/view/fiqa/",
        True,
        ["train", "dev", "test"],
        648,
        "57K",
        2.6,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "17918ed23cd04fb15047f73e6c3bd9d9",
    ),
    "arguana": DatasetInfo(
        "ArguAna",
        "http://argumentation.bplaced.net/arguana/data",
        True,
        ["test"],
        1406,
        "8.67K",
        1.0,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
        "8ad3e3c2a5867cdced806d6503f29b99",
    ),
    "webis-touche2020": DatasetInfo(
        "Touche-2020",
        "https://webis.de/events/touche-20/shared-task-1.html",
        True,
        ["test"],
        49,
        "382K",
        19.0,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
        "46f650ba5a527fc69e0a6521c5a23563",
    ),
    "cqadupstack": DatasetInfo(
        "CQADupstack",
        "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        True,
        ["test"],
        13145,
        "457K",
        1.4,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
        "4e41456d7df8ee7760a7f866133bda78",
    ),
    "quora": DatasetInfo(
        "Quora",
        "https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs",
        True,
        ["dev", "test"],
        10000,
        "523K",
        1.6,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
        "18fb154900ba42a600f84b839c173167",
    ),
    "dbpedia-entity": DatasetInfo(
        "DBpedia-Entity",
        "https://wiki.dbpedia.org/services-resources/datasets/dbpedia-version-2016-10",
        True,
        ["train", "test"],
        4672,
        "1.5M",
        2.1,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip",
        "1234567890abcdef1234567890abcdef",
    ),
    "scidocs": DatasetInfo(
        "SciDocs",
        "https://allenai.org/data/scidocs",
        True,
        ["train", "test"],
        18432,
        "2.9M",
        3.4,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
        "0987654321abcdef0987654321abcdef",
    ),
    "fever": DatasetInfo(
        "FEVER",
        "https://fever.ai",
        True,
        ["train", "dev", "test"],
        1454,
        "1.2M",
        2.7,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip",
        "abcdef1234567890abcdef1234567890",
    ),
    "climate-fever": DatasetInfo(
        "Climate-FEVER",
        "https://climatefever.ai",
        True,
        ["train", "test"],
        789,
        "345K",
        1.9,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip",
        "fedcba0987654321fedcba0987654321",
    ),
    "scifact": DatasetInfo(
        "SciFact",
        "https://allenai.org/data/scifact",
        True,
        ["train", "test"],
        2002,
        "680K",
        1.7,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "11223344556677881122334455667788",
    ),
    "germanquad": DatasetInfo(
        "GermanQuAD",
        "https://deepset.ai/germanquad",
        True,
        ["train", "test"],
        896,
        "421K",
        1.3,
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/germanquad.zip",
        "8899aabbccddeeff8899aabbccddeeff",
    ),
}


def download_raw(name, out_dir=None, overwrite=False) -> str:
    if name not in BEIR_DATASETS:
        raise ValueError(
            "Dataset {} not found. Available datasets: {}".format(name, BEIR_DATASETS.keys())
        )

    out_dir = out_dir or CF_HOME
    raw_dir = os.path.join(out_dir, "raw")
    output_path = os.path.join(raw_dir, name)

    # Check if the output directory already exists
    if os.path.exists(output_path):
        if overwrite:
            print("Output directory {} already exists. Overwriting.".format(output_path))
            shutil.rmtree(output_path)  # Remove the existing directory
        else:
            print("Output directory {} already exists. Skipping download.".format(output_path))
            return output_path

    os.makedirs(output_path, exist_ok=True)
    zip_file = os.path.join(out_dir, "{}.zip".format(name))
    url = BEIR_DATASETS[name].download_link

    print("Downloading {} ...".format(name))
    util.download_url(url, zip_file)

    print("Unzipping {} ...".format(name))
    util.unzip(zip_file, raw_dir)

    return output_path


def download_all(out_dir=None):
    for dataset in BEIR_DATASETS:
        download_raw(dataset, out_dir=out_dir)
