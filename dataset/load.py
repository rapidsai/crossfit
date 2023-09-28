from crossfit.dataset.beir import load as beir


def load_dataset(
    name,
    out_dir=None,
    blocksize=2**30,
    overwrite=False,
):
    if name.startswith("beir/"):
        return beir.load_dataset(
            name[len("beir/") :],
            out_dir=out_dir,
            blocksize=blocksize,
            overwrite=overwrite,
        )

    raise NotImplementedError(f"Unknown dataset: {name}")
