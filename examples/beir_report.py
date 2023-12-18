import argparse

import crossfit as cf


def parse_arguments():
    parser = argparse.ArgumentParser(description="BEIR evaluation using Crossfit")
    parser.add_argument("--pool-size", type=str, default="16GB", help="RMM pool size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")
    parser.add_argument(
        "--sorted-dataloader", default=True, action="store_true", help="Use sorted data loader"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--k", type=int, default=10, help="Nearest neighbors")
    parser.add_argument(
        "--partition-num",
        type=int,
        default=50_000,
        help="Number of items to allocate to each partition",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    vector_search = cf.TorchExactSearch(k=args.k)
    model = cf.SentenceTransformerModel(args.model_name)

    with cf.Distributed(rmm_pool_size=args.pool_size, n_workers=args.num_workers):
        report = cf.beir_report(
            dataset_name=args.dataset,
            model=model,
            vector_search=vector_search,
            overwrite=args.overwrite,
            sorted_data_loader=args.sorted_dataloader,
            batch_size=args.batch_size,
            partition_num=args.partition_num,
        )

    report.console()


if __name__ == "__main__":
    main()
