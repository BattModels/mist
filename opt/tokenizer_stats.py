#!/usr/bin/env python
import json
from pathlib import Path

import typer
from datasets import IterableDataset, load_dataset
from datasets.distributed import split_dataset_by_node
from electrolyte_fm.utils.tokenizer import load_tokenizer
from collections import Counter

cli = typer.Typer()


def is_invertable(input: str, output: str):
    input = input.strip()
    output = output.strip()
    if input == output:
        return True
    if output.startswith("<bos>"):
        output = output[5:]
    if output.endswith("<eos>"):
        output = output[:-5]
    return input == output


def usage_stats(batch, tokenizer, report_unk=False):
    codes = tokenizer(batch["text"])["input_ids"]

    # Tabulate token usage
    token_counts = Counter()
    fertility = Counter()
    nunique = Counter()
    noninvertable = 0
    for input, code in zip(batch["text"], codes):
        fertility[len(code)] += 1
        nunique[len(set(code))] += 1
        for token in code:
            token_counts[token] += 1

        output = tokenizer.decode(code)
        if not is_invertable(input, output):
            noninvertable += 1
            if report_unk:
                print(f"'{input}' != '{output}'")

    return {
        "token_usage": [token_counts],
        "fertility": [fertility],
        "nunique": [nunique],
        "samples": [len(codes)],
        "noninvertable": [noninvertable],
    }


def tabulate_dataset(dataset, tokenizer, rank=0):
    # Tokenize the dataset
    tokenized = dataset.map(
        usage_stats,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["text"],
        batch_size=1000,
    )
    out = dict()
    for i, batch in enumerate(tokenized):
        out = collate_results(out, batch)
        if i % 100 == 0:
            print(f"Rank {rank}: has processed {out['samples']} molecules")

    return out


def collate_results(out, batch):
    for k, v in batch.items():
        if k not in out:
            out[k] = v
        else:
            out[k] += v
    return out


@cli.command()
def mpi(tokenizer_path: str, path: Path, output_path: str = "stats.json"):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    tokenizer = load_tokenizer(tokenizer_path)

    # Get dataset of smiles
    print(f"Rank {rank} of {size} is starting")
    dataset = load_dataset(
        str(path.resolve()),
        split="train",
        streaming=True,
        keep_in_memory=False,
    )
    print(f"Rank {rank} of {size} finished loading dataset")
    dataset = split_dataset_by_node(dataset, rank, size)
    out = tabulate_dataset(dataset, tokenizer, rank)
    print(f"Rank {rank} has finished")
    out_collect = comm.gather(out, root=0)
    if rank == 0:
        print(f"Rank {rank}: Gathered results")
        out = dict()
        for rank_out in out_collect:
            out = collate_results(out, rank_out)
        with open(output_path, "w") as f:
            json.dump(out, f)


@cli.command()
def tabulate(
    tokenizer_path: str,
    path: Path,
    limit: int = None,
    output: str = "-",
):
    tokenizer = load_tokenizer(tokenizer_path)

    # Get dataset of smiles
    dataset = load_dataset(
        str(path.resolve()),
        split="train",
        streaming=True,
        keep_in_memory=False,
    )
    out = tabulate_dataset(dataset)

    with open("stats.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    cli()
