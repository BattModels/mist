from pathlib import Path
from tempfile import TemporaryDirectory

from test_fast_tokenizer import check_save, check_tokenize, check_unknown

from smirk import SmirkTokenizerFast
from smirk.cli import cli


def test_cli():
    with TemporaryDirectory() as output:
        # Train model
        cli([str(Path(__file__).parent.joinpath("opensmiles.smi")), "--output", output])
        tokenizer = SmirkTokenizerFast(
            tokenizer_file=Path(output).joinpath("tokenizer.json")
        )
        check_save(tokenizer)
        check_unknown(tokenizer)
        check_tokenize(tokenizer)
