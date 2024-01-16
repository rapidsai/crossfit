import pytest

cudf = pytest.importorskip("cudf")

import os  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from uuid import uuid4  # noqa: E402

from crossfit.dataset.load import load_dataset  # noqa: E402

examples_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "examples")


@pytest.mark.singlegpu
def test_beir_report():
    path = os.path.join(examples_dir, "beir_report.py")
    orig_sys_argv = sys.argv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "beir_report.py")
        shutil.copy2(path, tmp_path)
        # argv[0] will be replaced by runpy
        sys.argv = [
            "",
            "--overwrite",
            "--num-workers",
            "1",
            "--dataset",
            "fiqa",
            "--pool-size",
            "12GB",
            "--batch-size",
            "8",
            "--partition-num",
            "100",
        ]
        runpy.run_path(
            tmp_path,
            run_name="__main__",
        )

    sys.argv = orig_sys_argv


# Works locally (A6000) but does work in CI (P100)
@pytest.mark.singlegpu
def test_custom_pytorch_model():
    path = os.path.join(examples_dir, "custom_pytorch_model.py")
    orig_sys_argv = sys.argv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "custom_pytorch_model.py")
        shutil.copy2(path, tmp_path)

        dataset = load_dataset("beir/fiqa")
        dataset_path = os.path.join(tmpdir, f"{str(uuid4())}.parquet")
        dataset.item.ddf().sample(frac=0.01).to_parquet(dataset_path)

        output_path = os.path.join(tmpdir, f"{str(uuid4())}.parquet")

        # argv[0] will be replaced by runpy
        sys.argv = [
            "",
            f"{dataset_path}",
            f"{output_path}",
            "--pool-size",
            "4GB",
            "--batch-size",
            "8",
            "--partitions",
            "20",
        ]
        runpy.run_path(
            tmp_path,
            run_name="__main__",
        )

        df = cudf.read_parquet(output_path)
        labels = ["foo", "bar", "baz"]
        assert all(x in labels for x in df["labels"].unique().to_arrow().to_pylist())

    sys.argv = orig_sys_argv
