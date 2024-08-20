import argparse

from . import SmirkTokenizerFast


def cli(argv=None):
    p = argparse.ArgumentParser("python -m smirk.cli")
    p.add_argument("files", nargs="+")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument(
        "--merge-brackets", default=False, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "--split-structure", default=False, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "-o",
        "--output",
        default=".",
        type=str,
        help="directory where trained smirk-gpe model is saved",
    )
    args = p.parse_args(argv)
    tok = SmirkTokenizerFast()
    tok_gpe = tok.train(
        args.files,
        vocab_size=args.vocab_size,
        merge_brackets=args.merge_brackets,
        split_structure=args.split_structure,
    )
    tok_gpe.save_pretrained(args.output)


if __name__ == "__main__":
    cli()
