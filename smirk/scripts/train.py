import json
from smirk import SmirkTokenizerFast
from pathlib import Path
import argparse


def train(path, save_dir, limit=None, vocab_size=None):
    tok = SmirkTokenizerFast()
    files = [str(f) for f in Path(path).glob("*.txt")]
    if limit is not None:
        files = files[:limit]

    # Record training config
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir).joinpath("input_files.json"), "w") as fid:
        json.dump({"files": files, "vocab_size": vocab_size}, fid)

    trained = tok.train(files, vocab_size=vocab_size)
    trained.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to *.txt files to train on")
    parser.add_argument(
        "--limit", help="Limit the number of files", default=None, type=int
    )
    parser.add_argument(
        "-o", "--output", default="./smirk-gpe", help="Output directory", type=str
    )
    parser.add_argument("--vocab-size", default=None, type=int)
    args = parser.parse_args()
    train(args.path, args.output, vocab_size=args.vocab_size, limit=args.limit)
