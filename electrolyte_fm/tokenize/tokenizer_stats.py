#!/bin/bash
import json
from smirk import SmirkTokenizerFast
from pathlib import Path
import typer
import numpy as np
import pandas as pd
from pyspark import SparkContext
import pyspark.pandas as ps
from pyspark.sql import DataFrame, SparkSession
from collections import defaultdict

cli = typer.Typer()


def get_data(spark: SparkSession, directory: Path, avg_size=10e6):
    files = list(directory.glob("*.txt"))
    data = spark.read.text([str(x) for x in files])

    # Repartition to a target average size
    average_size = sum((f.stat().st_size for f in files)) / len(files)
    min_partition = int(np.ceil(average_size / avg_size))
    if data.rdd.getNumPartitions() < min_partition:
        data = data.repartition(min_partition)
    print(f"Using {data.rdd.getNumPartitions()} Partitions")
    return data


@cli.command()
def tokenizer_stats(
    tokenizer_path: str, path: Path, n_partitions=128, output: str = "-"
):
    tokenizer = SmirkTokenizerFast.from_pretrained(tokenizer_path)
    # tokenizer = SmirkTokenizerFast()
    sc = SparkContext()
    sc.setCheckpointDir(Path("/tmp").resolve().as_posix())
    spark = SparkSession(sc)

    # Get dataset of smiles
    data = get_data(spark, Path(path))

    # Init results
    results = dict()

    # Generate histogram of molecule lengths after being tokenized
    n_tokens = data.rdd.map(lambda s: len(tokenizer.encode_plus(s[0])["input_ids"]))
    parity, count = n_tokens.histogram(
        list(range(0, 2 + n_tokens.max()))
    )  # +2 to ensure that the last bucket contains the max value
    results["parity"] = dict(zip(parity, count))

    # Count the usage of each token
    results["token_counts"] = data.rdd.flatMap(
        lambda row: tokenizer.encode_plus(row[0])["input_ids"]
    ).countByValue()

    # Count the rate of OOV errors, returning examples iff found
    unk_token_id = tokenizer.unk_token_id
    oov = data.rdd.filter(
        lambda s: unk_token_id in tokenizer.encode_plus(s[0])["input_ids"]
    )
    results["oov_count"] = oov.count()
    results["oov_sample"] = [row[0] for row in oov.takeSample(False, 50)]
    results["exact_count"] = data.count()

    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if output == "-":
        print(json.dumps(results))
    else:
        with open(output, "w") as fid:
            json.dump(results, fid)


if __name__ == "__main__":
    cli()
