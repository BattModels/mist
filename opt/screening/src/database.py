import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from rdkit import Chem
from typer import Typer

from .utils import RateLimitedAdapter

# Sentinel for clean shutdown
_STOP = object()

cli = Typer()

logger = RateLimitedAdapter(logging.getLogger(__name__), min_interval=10)


def write_molecule_to_db(cursor: sqlite3.Cursor, mol: dict):
    smiles = mol["smi"]
    rdkit_mol = Chem.MolFromSmiles(smiles)
    if rdkit_mol is None:
        logger.warning("Invalid SMILES skipped: %s", smiles)
        return False

    inchi = Chem.MolToInchiKey(rdkit_mol)
    props_json = json.dumps({k: v for k, v in mol.items() if k != "smi"})

    cursor.execute(
        """
    INSERT INTO molecules (inchi, smiles, props, duplicate_count)
    VALUES (?, ?, ?, 1)
    ON CONFLICT(inchi) DO UPDATE SET
        smiles=excluded.smiles,
        props=excluded.props,
        duplicate_count=excluded.duplicate_count + 1
    """,
        (inchi, smiles, props_json),
    )
    return True


def _sqlite_writer(db_path, q: queue.Queue):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS molecules (
        inchi TEXT PRIMARY KEY,
        smiles TEXT,
        props TEXT,
        duplicate_count INTEGER DEFAULT 0
    );
    """)

    error_count = 0
    invalid_count = 0
    wab_count = 0
    wab_timeout = time.monotonic()
    try:
        while True:
            item = q.get()
            logger.debug(
                "queue_size: %d, errors: %d, wab_count: %d, invalid: %d",
                q.qsize(),
                error_count,
                wab_count,
                invalid_count,
            )
            if item is _STOP:
                break

            try:
                if not write_molecule_to_db(cursor, item):
                    invalid_count += 1

                # Periodically commit
                wab_count += 1
                if wab_count >= 50 or ((time.monotonic() - wab_timeout) > 10):
                    conn.commit()
                    wab_count = 0
                    wab_timeout = time.monotonic()

            except Exception as e:
                logger.error(f"Failed to insert molecule: {e}")
                error_count += 1

            finally:
                q.task_done()
    finally:
        try:
            conn.commit()
        except Exception:
            logger.error("Failed to commit transaction")
        conn.close()


def dump_to_sqlite_threaded(mol_iterator, db_path, queue_size=1024):
    """
    Writes molecules from an iterator to a SQLite DB asynchronously.

    Parameters:
    - mol_iterator (iterator): An iterator of dicts with at least a "smi" key.
    - db_path (str): Path to SQLite database.
    - queue_size (int): Max queue size before blocking producer.
    """
    logger.info("Writing molecules to %s", db_path)
    q = queue.Queue(maxsize=queue_size)
    writer_thread = threading.Thread(target=_sqlite_writer, args=(db_path, q))
    writer_thread.start()

    try:
        for mol in mol_iterator:
            q.put(mol)  # main thread only pushes raw dicts
    finally:
        q.put(_STOP)
        writer_thread.join()


@cli.command("merge-db")
def merge_sqlite_dbs(folder: Path, output_db: Optional[Path] = None):
    db_files = list(folder.glob("*.sqlite"))
    if not db_files:
        logger.warning("No .sqlite files found in %s", folder)
        return

    logger.info("Merging %d databases into %s", len(db_files), output_db)

    # Create or connect to the output DB
    output_db = output_db or folder.joinpath("merged.sqlite")
    output_db.unlink(missing_ok=True)
    if output_db in db_files:
        db_files.remove(output_db)
    conn_out = sqlite3.connect(output_db)
    cur_out = conn_out.cursor()
    logging.info("Merging: %s", ", ".join(str(f) for f in db_files))

    # Ensure schema exists
    cur_out.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            inchi TEXT PRIMARY KEY,
            smiles TEXT NOT NULL,
            props JSON,
            duplicate_count INTEGER DEFAULT 0
        );
    """)
    conn_out.commit()

    for db_path in db_files:
        logger.info("Merging %s", db_path.name)
        with sqlite3.connect(db_path) as conn_in:
            conn_in.row_factory = sqlite3.Row
            cur_in = conn_in.cursor()
            cur_in.execute("SELECT * FROM molecules")
            for row in cur_in:
                try:
                    # Attempt insert
                    cur_out.execute(
                        """
                        INSERT INTO molecules (inchi, smiles, props, duplicate_count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(inchi) DO UPDATE SET
                            duplicate_count = molecules.duplicate_count + 1
                    """,
                        (
                            row["inchi"],
                            row["smiles"],
                            row["props"],
                            row["duplicate_count"],
                        ),
                    )
                except Exception as e:
                    logger.error("Failed to merge row from %s: %s", db_path.name, e)
        conn_out.commit()

    # Track number of molecules and duplicates
    n_mols = cur_out.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
    n_dups = cur_out.execute("SELECT SUM(duplicate_count) FROM molecules").fetchone()[0]

    conn_out.close()
    logger.info(
        "Merge complete. %d unique molecules with %d duplicates", n_mols, n_dups
    )
