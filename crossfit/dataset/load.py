from crossfit.dataset.beir import load as beir


def load_dataset(name, out_dir=None, blocksize=2**30, overwrite=False, tiny_sample=False):
    load_fn_name = "load_dataset" if not tiny_sample else "load_test_dataset"

    if name.startswith("beir/"):
        return getattr(beir, load_fn_name)(
            name[len("beir/") :],
            out_dir=out_dir,
            blocksize=blocksize,
            overwrite=overwrite,
        )

    raise NotImplementedError(f"Unknown dataset: {name}")
