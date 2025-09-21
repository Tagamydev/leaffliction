from pathlib import Path
import polars as pl
from concurrent.futures import ThreadPoolExecutor


def _process_path(p: Path, exts: set) -> dict | None:
    """Helper function to process a single file path."""
    if p.is_file() and p.suffix.lower() in exts:
        return {
            "path": str(p.resolve()),
            "name": p.name,
            "class": p.parent.name,
            "stem": p.stem,
            "group": p.stem.split("_")[0],
        }
    return None


def list_images(root: str, max_workers: int = 8) -> pl.DataFrame:
    """
    Recursively scan for images under 'root' using multithreading.
    Returns a Polars DataFrame with metadata.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    paths = list(Path(root).rglob("*"))

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(lambda p: _process_path(p, exts), paths):
            if result:
                rows.append(result)

    return pl.from_records(rows)


def train_test_val(df: pl.DataFrame) -> pl.DataFrame:
    """Create a train/val/test split based on group."""
    TRAIN, VAL = (0.7, 0.9)

    df2 = df.select(["class", "group"]).unique().sample(fraction=1, shuffle=True)

    df3 = df2.with_columns(
        idx=pl.cum_count("group").over("class"),
        tot=pl.n_unique("group").over("class"),
    )

    df4 = df3.with_columns(
        prop=pl.col("idx") / pl.col("tot")
    )

    df5 = df4.with_columns(
        split=(
            pl.when(pl.col("prop") < TRAIN).then(pl.lit("train"))
            .when(pl.col("prop") < VAL).then(pl.lit("val"))
            .otherwise(pl.lit("test"))
        )
    )

    df5 = df5.drop(["idx", "tot", "prop", "class"])
    return df5


def build_dataset_csv(root: str, out_csv: str, max_workers: int = 8) -> None:
    """
    Build dataset CSV with metadata and splits.
    """
    x = list_images(root, max_workers=max_workers)
    y = train_test_val(x)
    z = x.join(y, on=["group"])
    z.write_csv(out_csv)
    print(f"[INFO] Dataset CSV saved to {out_csv}")

