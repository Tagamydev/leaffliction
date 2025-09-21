import argparse
import os
import multiprocessing
from srcs.Distribution.dataset_builder import build_dataset_csv
from srcs.Distribution.visualize import visualize_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Scan images, build dataset CSV, and visualize class distribution."
    )

    parser.add_argument(
        "directory",
        type=str,
        help="Root directory containing images"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dataset.csv",
        help="Output CSV file name (default: dataset.csv)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of threads to use (default: max available)"
    )
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only build dataset CSV, do not generate graphs"
    )
    parser.add_argument(
        "--graph_only",
        action="store_true",
        help="Only generate graphs from existing CSV (skip building)"
    )

    args = parser.parse_args()

    if args.build_only and args.graph_only:
        parser.error("Options --build_only and --graph_only cannot be used together.")

    csv_path = args.output

    if args.graph_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[ERROR] CSV file '{csv_path}' not found. Run build first.")
        visualize_dataset(csv_path)
        return

    # Always build unless graph_only is set
    build_dataset_csv(args.directory, csv_path, max_workers=args.threads)

    if not args.build_only:
        visualize_dataset(csv_path)


if __name__ == "__main__":
    main()
