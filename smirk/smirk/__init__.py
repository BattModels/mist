# Import Rust Binding
from .smirk import *
from pathlib import Path

VOCAB_FILE = Path(__file__).parent.parent.joinpath("vocab.json")
