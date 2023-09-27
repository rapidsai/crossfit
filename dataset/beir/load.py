import os

from crossfit.dataset.home import CF_HOME
from crossfit.dataset.base import IRDataset
from crossfit.dataset.beir.raw import download_raw


def load_dataset(
    name,
    out_dir=None,
    blocksize=2**30,
    overwrite=False,
    # batch_size: int = 1024,
    part_size="300MB",
) -> IRDataset:
    raw_path = download_raw(name, out_dir=out_dir, overwrite=False)

    out_dir = out_dir or CF_HOME
    processed_dir = os.path.join(out_dir, "processed", name)

    if os.path.exists(processed_dir):
        return IRDataset.from_dir(processed_dir)
