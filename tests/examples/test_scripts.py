import os
import runpy
import shutil
import sys
import tempfile

import pytest

examples_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "examples")


@pytest.mark.singlegpu
@pytest.mark.parametrize(
    "script",
    [
        "beir_report.py",
    ],
)
def test_script_execution(script):
    path = os.path.join(examples_dir, script)
    orig_sys_argv = sys.argv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, script)
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
