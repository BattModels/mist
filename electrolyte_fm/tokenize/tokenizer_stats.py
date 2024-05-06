#!/bin/bash
import json
from smirk import SmirkTokenizerFast
from pathlib import Path
import typer
import numpy as np
from pyspark.sql import SparkSession
from transformers import PreTrainedTokenizerBase, AutoTokenizer

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


def is_oov(input: str, tokenizer: PreTrainedTokenizerBase):
    code = tokenizer.encode_plus(input)["input_ids"]
    if tokenizer.unk_token_id in code:
        print("found unk")
        return True
    # Check if the tokenizer *really* encoded the full string
    if tokenizer.decode(code) != input:
        print("decode failed")
        return True
    return False


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    if name.startswith("smirk"):
        if name == "smirk":
            return SmirkTokenizerFast()
        else:
            return SmirkTokenizerFast.from_pretrained(name)
    return AutoTokenizer.from_pretrained(
        name, cache_dir=".cache", trust_remote_code=True
    )


@cli.command()
def tokenizer_stats(
    tokenizer_path: str, path: Path, n_partitions=128, output: str = "-"
):
    spark = SparkSession.builder.getOrCreate()
    tokenizer = load_tokenizer(tokenizer_path)

    # Get dataset of smiles
    path = Path(path).resolve()
    data = get_data(spark, path)

    # Init results
    results = dict()

    # Generate histogram of molecule lengths after being tokenized
    n_tokens = data.rdd.map(lambda s: len(tokenizer(s[0])["input_ids"]))
    parity, count = n_tokens.histogram(
        list(range(0, 2 + n_tokens.max()))
    )  # +2 to ensure that the last bucket contains the max value
    results["parity"] = dict(zip(parity, count))

    # Count the usage of each token
    results["token_counts"] = data.rdd.flatMap(
        lambda row: tokenizer.encode_plus(row[0])["input_ids"]
    ).countByValue()

    # Count the rate of OOV errors, returning examples iff found
    oov = data.rdd.filter(lambda s: is_oov(s[0], tokenizer))
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

    # Save Tokenizer
    tokenizer.save_pretrained(Path(output).parent)


if __name__ == "__main__":
    cli()
