import argparse
import logging
import os
import random
import re
import select
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from datasets import IterableDataset
from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from electrolyte_fm.data_modules.utils import MolEncoding

logger = logging.getLogger(__name__)


def stream_fasmifra(
    fragments_file: str,
    exe_path: Path = Path(__file__).parent.parent.joinpath("vendor", "fasmifra.exe"),
    seed: int = 42,
    read_timeout: float = 60.0,
):
    with TemporaryDirectory() as tmpdir:
        fifo_id = random.randint(0, 65_535)
        fifo_path = Path(tmpdir, f"fasmifra_fifo_{fifo_id}.smi")
        rm_cut_bonds = re.compile(r"\[\*:\d+\]\[\*:\d+\]")

        # Create named pipe
        os.mkfifo(fifo_path)
        logger.info("Created fasmifra FIFO at %s", fifo_path)

        # Define the reader function using non-blocking I/O and select.
        def reader():
            try:
                # Open FIFO in nonblocking mode.
                fd = os.open(str(fifo_path), os.O_RDONLY | os.O_NONBLOCK)
                # Wrap the FD into a file object.
                with os.fdopen(fd, "r", buffering=1) as fifo:
                    # Continue reading until the stop condition
                    while True:
                        # Use select to wait for data with timeout.
                        ready, _, _ = select.select([fifo], [], [], read_timeout)
                        if ready:
                            line = fifo.readline()
                            if line == "":  # EOF reached
                                break
                            yield rm_cut_bonds.sub("", line.rstrip().split("\t")[0])

                        else:
                            logger.error("timed out waiting for data on %s", fifo_path)
                            break

            except Exception as e:
                logger.exception("Error reading FIFO: %s", e)

        # Launch fasmifra as a subprocess.
        proc = subprocess.Popen(
            [
                str(exe_path),
                "-i",
                str(fragments_file),
                "-n",
                str(sys.maxsize // 2),
                "--seed",
                str(seed),
                "-o",
                str(fifo_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            # Yield items from the queue.
            yield from reader()

            # Once done, wait for the process.
            proc.terminate()
            retcode = proc.wait()
            if retcode != 0:
                stderr_output = (
                    proc.stderr.read().decode("utf-8") if proc.stderr else "(no stderr)"
                )
                logger.error("fasmifra failed with return code %d", retcode)
                raise RuntimeError(
                    f"fasmifra.exe crashed with return code {retcode}: {stderr_output}"
                )
        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                fifo_path.unlink(missing_ok=True)
            except Exception:
                pass


def shard_fragments(fabric: Fabric, ref_fragments: str | Path) -> Path:
    lines = None
    if fabric.is_global_zero:
        lines = Path(ref_fragments).read_text().split("\n")
        random.shuffle(lines)
    fabric.barrier("shard_fragments - broadcast")
    lines = fabric.broadcast(lines)
    assert isinstance(lines, list)

    # Slice the data for this rank
    lines = lines[fabric.global_rank :: fabric.world_size]
    logger.info(
        "rank %d: got %d fragments from %s",
        fabric.global_rank,
        len(lines),
        ref_fragments,
    )

    # Write lines to disk
    fid = NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=f"_rank_{fabric.global_rank}.smi",
        delete=False,
    )
    try:
        for line in lines:
            fid.write(line + "\n")
        fid.flush()
        fid.close()
    except Exception:
        Path(fid.name).unlink(missing_ok=True)

    return Path(fid.name)


def shard_with_replacement(fabric: Fabric, ref_fragments: str | Path) -> Path:
    p_pick = 1 / fabric.world_size
    fid = NamedTemporaryFile(
        "w", encoding="utf-8", suffix=f"_rank_{fabric.global_rank}.smi", delete=False
    )
    with open(ref_fragments, "r") as frags:
        for line in frags:
            if random.random() < p_pick:
                fid.write(line)

    return Path(fid.name)


def dataloader(
    ref_fragments: str,
    tokenizer,
    batch_size: int = 512,
    encoding: str | None = None,
    fabric: Fabric | None = None,
):
    def gen(ref_frag_file: str | Path):
        unlink_frag_file = False
        try:
            if fabric is not None:
                ref_frag_file = shard_with_replacement(fabric, ref_frag_file)
                unlink_frag_file = True

            logger.info("using fragments from %s", ref_frag_file)
            encoder = MolEncoding(encoding or "smiles")
            for smi in stream_fasmifra(str(ref_frag_file)):
                yield {"smi": encoder(smi)}

        finally:
            if unlink_frag_file:
                assert isinstance(ref_frag_file, Path)
                ref_frag_file.unlink(missing_ok=True)

    ds = IterableDataset.from_generator(
        gen, gen_kwargs={"ref_frag_file": ref_fragments}
    )
    token_collate = DataCollatorWithPadding(tokenizer)

    def collate(batch):
        smi = [x.pop("smi") for x in batch]
        out = token_collate(batch)
        out["smi"] = smi
        return out

    ds = ds.map(tokenizer, input_columns="smi", batched=True)
    return DataLoader(
        ds,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=1,
        prefetch_factor=2,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    for mol in stream_fasmifra(args.input):
        print(mol)
