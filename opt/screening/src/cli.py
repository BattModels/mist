import logging

import typer

from .database import merge_sqlite_dbs
from .fasmifra import stream_fasmifra

logging.basicConfig(level=logging.INFO)

cli = typer.Typer()
cli.command("merge-db")(merge_sqlite_dbs)
cli.command("fasmifra")(stream_fasmifra)


if __name__ == "__main__":
    cli()
