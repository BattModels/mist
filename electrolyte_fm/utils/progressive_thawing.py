import logging
from fnmatch import fnmatch
from lightning.pytorch.callbacks import BaseFinetuning


class ProgressiveThawing(BaseFinetuning):
    def __init__(
        self,
        initial: list[str],
        stages: list[list[str]],
        stage_duration: int = 1,
        lr_initial: float = 10,
        lr_decay: float = 0.99,
    ):
        super().__init__()
        self.initial = initial
        self.stages = stages
        self.stage_duration = stage_duration
        self.current_stage = 0
        self.lr_scale = [0]
        self.lr_initial = float(lr_initial)
        self.lr_decay = float(lr_decay)

    def setup(self, trainer, pl_module, stage) -> None:
        if not hasattr(pl_module, "encoder"):
            pl_module.configure_model()
        self.freeze_before_training(pl_module)

    @classmethod
    def matching_modules(cls, pl_module, patterns):
        for pattern in patterns:
            pattern_matched = False
            for name, module in pl_module.named_modules():
                if name == "":
                    continue
                logging.debug(
                    "Matching %s against %s: %d", name, pattern, fnmatch(name, pattern)
                )
                if fnmatch(name, pattern):
                    pattern_matched = True
                    yield name, module

            if not pattern_matched:
                names = [name for name, _ in pl_module.named_modules()]
                raise ValueError(f"Pattern {pattern} not found in module: {names}")

    def unfreeze_and_add_param_group(self, modules, optimizer):
        self.make_trainable(modules)
        params = self.filter_params(modules)
        params = self.filter_on_optimizer(optimizer, params)
        base_lr = optimizer.param_groups[0]["lr"]
        if params:
            self.lr_scale.append(self.lr_initial)
            lr_scale = self.lr_scale[-1] + 1
            optimizer.add_param_group({"params": params, "lr": base_lr / lr_scale})

    def on_train_batch_start(self, trainer, *args, **kwargs):
        # Decay lr offset
        for idx in range(len(self.lr_scale)):
            self.lr_scale[idx] *= self.lr_decay

        for opt in trainer.optimizers:
            base_lr = opt.param_groups[0]["lr"]
            for idx, param_group in enumerate(opt.param_groups):
                param_group["lr"] = base_lr / (self.lr_scale[idx] + 1)

    def lr_scheduler_step(self, scheduler, metric) -> None:
        print("LR Scheduler step", scheduler, metric)

    def freeze_before_training(self, pl_module):
        for name, module in self.matching_modules(pl_module, self.initial):
            logging.info("Freezing %s", name)
            self.freeze_module(module)

    def finetune_function(self, pl_module, epoch: int, optimizer) -> None:
        if self.current_stage == len(self.stages):
            return

        if epoch == self.stage_duration * (self.current_stage + 1):
            thaw = self.stages[self.current_stage]
            self.current_stage += 1
            to_thaw = []
            for name, module in self.matching_modules(pl_module, thaw):
                logging.info("Thawing %s", name)
                to_thaw.append(module)
            self.unfreeze_and_add_param_group(to_thaw, optimizer)
