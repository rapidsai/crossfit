from crossfit.dataset.beir import load as beir


def load_dataset(
    name,
    out_dir=None,
    blocksize=2**30,
    overwrite=False,
    # batch_size: int = 1024,
    part_size="300MB",
):
    if name.startswith("beir/"):
        return beir.load_dataset(
            name[len("beir/") :],
            out_dir=out_dir,
            blocksize=blocksize,
            overwrite=overwrite,
            part_size=part_size,
            # batch_size=batch_size,
        )

    raise NotImplementedError(f"Unknown dataset: {name}")
