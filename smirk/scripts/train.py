from smirk import SmirkTokenizerFast
from pathlib import Path
import argparse


def train(path, save_dir, vocab_size=None):
    tok = SmirkTokenizerFast()
    files = [str(f) for f in Path(path).glob("*.txt")]
    trained = tok.train(files, vocab_size=vocab_size)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trained.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to *.txt files to train on")
    parser.add_argument(
        "-o", "--output", default="./smirk-gpe", help="Output directory", type=str
    )
    parser.add_argument("--vocab-size", default=None, type=int)
    args = parser.parse_args()
    train(args.path, args.output, vocab_size=args.vocab_size)
