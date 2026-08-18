#!/usr/bin/env python3
"""Prepare pinned SVAMP or ARC-Easy data in the VERL prompt schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_dataset_screen import DATASETS, adapt_records, file_sha256, validate_held_out


def prepare(dataset: str, output_root: Path) -> dict:
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = DATASETS[dataset]
    kwargs = {"path": spec.repository, "revision": spec.revision}
    if spec.config is not None:
        kwargs["name"] = spec.config
    raw = load_dataset(**kwargs)
    train = adapt_records(dataset, raw[spec.train_split], "train")
    validation = adapt_records(dataset, raw[spec.validation_split], "validation")
    if len(train) != spec.expected_train_rows or len(validation) != spec.expected_validation_rows:
        raise RuntimeError(
            f"row count mismatch for {dataset}: "
            f"{len(train)}/{len(validation)} != "
            f"{spec.expected_train_rows}/{spec.expected_validation_rows}"
        )
    validate_held_out(train, validation)
    output_root.mkdir(parents=True, exist_ok=False)
    files = {}
    for filename, rows in (("train.parquet", train), ("validation.parquet", validation)):
        path = output_root / filename
        pq.write_table(
            pa.Table.from_pylist(rows),
            path,
            compression="zstd",
            version="2.6",
            row_group_size=1024,
            write_statistics=True,
        )
        files[filename] = {"rows": len(rows), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "data_source": spec.source,
        "repository": spec.repository,
        "config": spec.config,
        "revision": spec.revision,
        "seed": 0,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
        "files": files,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
