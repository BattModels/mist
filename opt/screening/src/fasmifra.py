import argparse
import logging
import os
import random
import re
import select
import subprocess
import sys
from typing import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile


logger = logging.getLogger(__name__)


def stream_fasmifra(
    fragments_file: str,
    exe_path: Path = Path(__file__).parent.parent.joinpath("vendor", "fasmifra"),
    seed: int = 42,
    read_timeout: float = 60.0,
):
    with TemporaryDirectory() as tmpdir:
        fifo_id = random.randint(0, 65_535)
        fifo_path = Path(tmpdir, f"fasmifra_fifo_{fifo_id}.smi")
        rm_cut_bonds = re.compile(r"\[\*:\d+\]\[\*:\d+\]")

        # Create named pipe
        os.mkfifo(fifo_path)
        logger.debug("Created fasmifra FIFO at %s for %s", fifo_path, fragments_file)

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


class FASMIFRA:
    def __init__(self, fragments: Iterable[str], encoder=None, **kwargs):
        self.fragment_file = NamedTemporaryFile("w", encoding="utf-8")
        n_frags = 0
        for frag in fragments:
            n_frags += 1
            self.fragment_file.write(frag + "\n")
        logging.debug(
            "setting up fasmifra with %d fragments at %s",
            n_frags,
            self.fragment_file.name,
        )
        self.fragment_file.flush()
        self.encoder = encoder or (lambda x: x)
        self.kwargs = kwargs
        self._iterator = None

    def __call__(self):
        return self

    def __iter__(self):
        for smi in stream_fasmifra(self.fragment_file.name, **self.kwargs):
            yield {"smi": self.encoder(smi)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    for mol in stream_fasmifra(args.input):
        print(mol)
