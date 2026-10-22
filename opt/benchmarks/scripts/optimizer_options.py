"""Optimizer arguments shared by the benchmark sweep entrypoints."""

import argparse


def beta(value):
    value = float(value)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("betas must be in [0, 1)")
    return value


def warmup_steps(value):
    if value == "beta2":
        return value
    try:
        steps = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("warmup must be beta2 or a nonnegative integer") from None
    if steps < 0:
        raise argparse.ArgumentTypeError("warmup steps must be nonnegative")
    return steps


def add_optimizer_options(parser, *, warmup_default):
    parser.add_argument(
        "--betas", type=beta, nargs=2, default=(0.9, 0.999),
        metavar=("BETA1", "BETA2"), help="AdamW momentum and variance decay",
    )
    parser.add_argument(
        "--warmup-steps", type=warmup_steps, default=warmup_default,
        help="Warmup optimizer steps, or beta2 for ceil(2 / (1 - beta2))",
    )
