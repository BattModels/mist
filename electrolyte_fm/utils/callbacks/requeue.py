import os
import sys
import signal
import logging
from subprocess import PIPE, run, Popen, STDOUT
from pathlib import Path
from shutil import which
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning import Callback

from ..ckpt import SaveConfigWithCkpts

_SIGNUM = Union[int, signal.Signals]

log = logging.getLogger(__name__)


class Requeue(Callback):
    def __init__(self, requeue_signal: Optional[signal.Signals] = None):
        requeue_signal = requeue_signal or signal.SIGTERM
        self.signal = signal.Signals(requeue_signal)
        self.ckpt_path: Optional[Path] = None
        self.requeue_count: int = 0
        super().__init__()

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.trainer = trainer
        self.ckpt_path = SaveConfigWithCkpts.log_dir(trainer).joinpath("checkpoints")
        signal.signal(self.signal, self.handle_requeue_signal)
        log.info("Registered handler for %s", self.signal.name)

    def handle_requeue_signal(self, signum, frame):
        # Save a checkpoint
        assert self.ckpt_path is not None
        self.requeue_count += 1
        ckpt_path = self.ckpt_path.joinpath("requeue.ckpt")
        self.trainer.save_checkpoint(ckpt_path)

        if self.trainer.is_global_zero:
            log.info("Requeuing using %s", ckpt_path)
            self.requeue(ckpt_path)

    def requeue(self, ckpt_path: Path):
        if which("scontrol"):
            self._requeue_slurm(ckpt_path)
        if which("qrerun"):
            self._requeue_pbs(ckpt_path)
        else:
            raise RuntimeError("Unknown HPC Environment -> Failed to requeue")

        # Exit
        exit(0)

    def _requeue_slurm(self, ckpt_path: Path):
        if job_array_id := os.environ.get("SLURM_ARRAY_JOB_ID", None):
            job_id = job_array_id + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
        else:
            job_id = os.environ["SLURM_JOB_ID"]

        requeue_marker = ckpt_path.parent.parent.joinpath("requeue", job_id)
        requeue_marker.parent.mkdir(exist_ok=True, parents=True)
        requeue_marker.unlink(missing_ok=True)
        os.symlink(ckpt_path.resolve(), requeue_marker)

        run(["scontrol", "requeue", job_id], check=True)

    def _requeue_pbs(self, ckpt_path: Path):
        if job_array_id := os.environ.get("PBS_ARRAY_ID", None):
            job_id = job_array_id + "_" + os.environ["PBS_ARRAY_INDEX"]
        else:
            job_id = os.environ["PBS_JOBID"]

        requeue_marker = ckpt_path.parent.parent.joinpath("requeue", job_id)
        requeue_marker.parent.mkdir(exist_ok=True, parents=True)
        requeue_marker.unlink(missing_ok=True)
        os.symlink(ckpt_path.resolve(), requeue_marker)

        script = Popen(
            ["submit/resubmit", "submit/polaris.j2", ckpt_path, "--resume-wandb"],
            stdout=PIPE,
            stderr=STDOUT,
        )
        submit = Popen(["qsub"], stdin=script.stdout, stdout=PIPE, stderr=STDOUT)
        script.stdout.close()

        output, errors = submit.communicate()

        log.info(output.decode())
        if errors:
            log.error(errors.decode())

    def state_dict(self):
        return {
            "signal": self.signal.numerator,
            "ckpt_path": self.ckpt_path,
            "requeue_count": self.requeue_count,
        }

    def load_state_dict(self, state_dict):
        self.signal = signal.Signals(state_dict["signal"])
        self.ckpt_path = state_dict["ckpt_path"]
        self.requeue_count = state_dict["requeue_count"]
