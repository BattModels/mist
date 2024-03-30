from pathlib import Path
from jsonargparse import Namespace
import json

from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning import Trainer, LightningModule, Callback


class SaveConfigWithCkpts(Callback):
    """Save Configuration with the model's checkpoints"""

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        overwrite: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.overwrite = overwrite
        self.already_saved = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        log_dir = trainer.log_dir or Path.cwd()
        if logger := trainer.logger:
            config_path = Path(log_dir, str(logger.name), str(logger.version))
        else:
            config_path = Path(log_dir)

        if trainer.is_global_zero:
            config_path.mkdir(parents=True, exist_ok=True)
            self.parser.save(
                self.config,
                Path(config_path, "config.json"),
                skip_none=False,
                overwrite=self.overwrite,
                format="json",
            )

            # Save model hyperparameters
            with open(Path(config_path, "model_hparams.json"), "w") as fid:
                json.dump(trainer.lightning_module.hparams, fid)

            if logger := trainer.logger:
                logger.log_hyperparams({"cli": self.config.as_dict()})
