"""Fragment ZINC, merge its fragments, and publish a database ready for sampling."""

import argparse
import logging
import os
import shutil
import sqlite3
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fragment import fragment_smi, store_fragments_in_sqlite

logger = logging.getLogger(__name__)


def copy_database(source, destination):
    """Publish a complete copy without exposing an interrupted copy as a database."""
    pending = destination.with_suffix(".sqlite.partial")
    try:
        shutil.copyfile(source, pending)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)


def stage_database(checkpoint, directory):
    primary = directory / checkpoint.name
    if not primary.exists():
        copy_database(checkpoint, primary)
        logger.info("Staged %s in %s", checkpoint.name, directory)
    return primary


def finalize_database(source, output):
    """Give the merged fragments dense row IDs before publishing the final output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".sqlite.partial")
    pending.unlink(missing_ok=True)
    logger.info("Materializing sampling database %s from %s", output, source)
    try:
        with closing(sqlite3.connect(pending, uri=True)) as database, database:
            database.execute(
                "ATTACH DATABASE ? AS source",
                (source.resolve().as_uri() + "?mode=ro",),
            )
            # The source already contains unique fragments. An ordinary table
            # supplies dense rowids without duplicating the text in an index.
            database.execute(
                "CREATE TABLE fragments AS SELECT fragment FROM source.fragments"
            )
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    if source != output:
        source.unlink()
    logger.info("Created sampling database %s", output)
    return output


def merge_pair(lhs, rhs, directory):
    primary_lhs = stage_database(lhs, directory)
    primary_rhs = stage_database(rhs, directory)
    with closing(sqlite3.connect(primary_lhs)) as database, database:
        database.execute("PRAGMA main.journal_mode = DELETE")
        database.execute("PRAGMA main.cache_size = -2097152")
        database.execute("ATTACH DATABASE ? AS rhs", (str(primary_rhs),))
        database.execute(
            "INSERT OR IGNORE INTO main.fragments (fragment) "
            "SELECT fragment FROM rhs.fragments "
            "ORDER BY fragment"
        )
    # Publish the new checkpoint before removing either RHS copy.
    # If interrupted here, retrying with both checkpoints is safe: inserts deduplicate.
    copy_database(primary_lhs, lhs)
    rhs.unlink()
    primary_rhs.unlink()
    logger.info("Merged and checkpointed %s <- %s", lhs.name, rhs.name)
    return lhs


def merge_fragments(directory, output, workers=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    databases = sorted(path for path in directory.glob("*.sqlite") if path != output)
    if not databases:
        raise ValueError(f"No input databases in {directory}")
    # An interrupted checkpoint copy is never a usable restart input.
    for partial in directory.glob("*.sqlite.partial"):
        partial.unlink()
    workers = workers if workers is not None else len(os.sched_getaffinity(0))
    workers = min(workers, max(1, len(databases) // 2))
    logger.info("Merging fragment databases with %d worker processes", workers)
    ready = deque(databases)
    remaining = len(databases)
    pending = set()
    # Each allocation starts on a fresh node. Only .cache survives a requeue;
    # within this run all workers share and reuse the same /tmp primaries.
    with (
        TemporaryDirectory(prefix="mist-fragment-merge-", dir="/tmp") as temporary,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        primary_directory = Path(temporary)
        logger.info("Using local merge workspace %s", primary_directory)
        try:
            while pending or len(ready) > 1:
                while len(ready) > 1 and len(pending) < workers:
                    pending.add(
                        pool.submit(
                            merge_pair,
                            ready.popleft(),
                            ready.popleft(),
                            primary_directory,
                        )
                    )
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    # A result is reusable only after its checkpoint is published
                    # and its RHS is deleted, so no database has two writers.
                    ready.append(future.result())
                    remaining -= 1
                    logger.info("%d databases remain", remaining)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return finalize_database(ready.pop(), output)


def fragment_shard(source, directory, databases, fragmenter):
    directory.mkdir(parents=True, exist_ok=True)
    fragments = directory / "fragments.frag"
    if not fragments.is_file():
        pending = fragments.with_suffix(".frag.partial")
        fragment_smi(
            source, str(pending), fragmenter=fragmenter, interpreter=sys.executable
        )
        pending.replace(fragments)
    database = databases / f"{directory.name}.sqlite"
    if not database.is_file():
        store_fragments_in_sqlite(directory)
        (directory / "fragments.sqlite").replace(database)


def fragment_zinc(zinc, fragmenter, work_dir, output, workers=None):
    zinc, fragmenter = Path(zinc), Path(fragmenter)
    work_dir, output = Path(work_dir), Path(output)
    if output.is_file():
        with closing(sqlite3.connect(output)) as database:
            table = database.execute(
                "SELECT wr FROM pragma_table_list "
                "WHERE schema = 'main' AND name = 'fragments'"
            ).fetchone()
        if table is None:
            raise ValueError(f"No fragments table in {output}")
        if not table[0]:
            return output
        # Upgrade outputs made before finalization supplied sampling rowids.
        return finalize_database(output, output)
    workers = workers if workers is not None else len(os.sched_getaffinity(0))
    merge_dir = work_dir / "merge"
    if not merge_dir.exists():
        sources = sorted(zinc.rglob("*.txt"))
        if not sources:
            raise ValueError(f"No ZINC text shards beneath {zinc}")
        imports = work_dir / "imports"
        imports.mkdir(parents=True, exist_ok=True)
        # Input conversion is Python work, so give each shard its own interpreter.
        logger.info("Processing ZINC shards with %d worker processes", workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    fragment_shard,
                    source,
                    work_dir / "fragments" / f"{index:06d}",
                    imports,
                    fragmenter,
                )
                for index, source in enumerate(sources)
            ]
            try:
                for completed, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    logger.info("Imported %d/%d ZINC shards", completed, len(sources))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        # Only start merging once every shard has been imported. On a retry,
        # merge the surviving databases without recreating deleted RHS inputs.
        imports.rename(merge_dir)
    return merge_fragments(merge_dir, output, workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zinc", type=Path, required=True)
    parser.add_argument("--fragmenter", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    fragment_zinc(args.zinc, args.fragmenter, args.work_dir, args.output, args.workers)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    main()
