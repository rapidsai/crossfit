import argparse

import crossfit as cf
from crossfit.dataset.load import load_dataset


def main():
    args = parse_arguments()

    dataset = load_dataset(args.dataset, tiny_sample=args.tiny_sample)
    query = dataset.query.ddf()

    with cf.HFGenerator(args.path_or_name, num_gpus=args.num_gpus) as generator:
        results = generator.infer(query, col="text")

    results.to_parquet(args.output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate text using CrossFit")
    parser.add_argument("path_or_name")
    parser.add_argument(
        "--dataset", default="beir/fiqa", help="Dataset to load (default: beir/fiqa)"
    )
    parser.add_argument(
        "--tiny-sample", default=True, action="store_true", help="Use tiny sample dataset"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2, help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--output-file",
        default="generated_text.parquet",
        help="Output Parquet file (default: generated_text.parquet)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
