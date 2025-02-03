#!/usr/bin/env python
from pathlib import Path

import numpy as np
import typer
from pyspark import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

cli = typer.Typer()

data_split = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def hashSplit(df: DataFrame, split: list[float], mod: int = 10000) -> list[DataFrame]:
    norm_weights = mod * np.cumsum(split) / np.sum(split)
    cuttoff = [w * 2 - mod for w in norm_weights]
    idf = df.withColumn("__hash", sf.xxhash64("value") % mod)
    splits = []
    low = -mod
    hash_col = sf.col("__hash")
    for high in cuttoff:
        split = idf.filter(hash_col >= low).filter(hash_col < high).drop(hash_col)
        splits.append(split)
        low = high
    return splits


@cli.command()
def split(
    path: Path,
    dataset_path: Path,
    n: int = typer.Option(1024, help="Shard dataset into n paritions"),
):
    print("starting split")
    spark = SparkSession(SparkContext())
    print("got spark")
    data = spark.read.text(str(path.joinpath("*.txt")))
    splits = hashSplit(data, [x for x in data_split.values()])
    for name, split in zip(data_split.keys(), splits):
        # Ensure we have exactly n partitions
        if split.rdd.getNumPartitions() < n:
            split = split.repartition(n)
        else:
            split = split.coalesce(n)

        # Write out dataset
        split.write.mode("overwrite").text(str(dataset_path.joinpath("data", name)))


@cli.command()
def shard(path: Path, dataset_path: Path):
    spark = SparkSession(SparkContext())
    data = spark.read.text(str(path.joinpath("*.txt")))
    data.write.mode("overwrite").text(str(dataset_path.joinpath("data", "raw")))


if __name__ == "__main__":
    print("starting cli")
    cli()
