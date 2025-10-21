import json
import torch
from lightning.pytorch.cli import (
    LightningCLI,
    LightningArgumentParser,
    _InstantiatorFn,
    _get_module_type,
)


def automodel_from_pretrained(
    name_or_path: str, trust_remote_code=False, add_pooling_layer=None
) -> torch.nn.Module:
    from transformers import AutoModel

    kwargs = {}
    if add_pooling_layer is not None:
        kwargs["add_pooling_layer"] = add_pooling_layer

    return AutoModel.from_pretrained(
        name_or_path, trust_remote_code=trust_remote_code, **kwargs
    )


def mlm_from_pretrained(
    name_or_path: str, trust_remote_code: bool = True
) -> torch.nn.Module:
    from transformers import AutoModelForMaskedLM

    return AutoModelForMaskedLM.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )


def recursive_update(original, updates):
    """
    Recursively updates a dictionary with another dictionary.
    """
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            # If both original and updates have a dictionary at this key, recurse
            recursive_update(original[key], value)
        else:
            # Otherwise, update the value directly
            original[key] = value
    return original


class MistLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("parser_kwargs", {"parser_mode": "jsonnet"})
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument(
            "--tags",
            type=list,
            help="Tags for WandB logger",
            default=[],
        )
        parser.link_arguments("tags", "trainer.logger.init_args.tags")

    def _add_instantiators(self) -> None:
        self.config_dump = json.loads(
            self.parser.dump(
                self.config, skip_link_targets=False, skip_none=False, format="json"
            )
        )
        if "subcommand" in self.config:
            self.config_dump = self.config_dump[self.config.subcommand]

        self.parser.add_instantiator(
            _InstantiatorFn(cli=self, key="model"),
            _get_module_type(self._model_class),
            subclasses=self.subclass_mode_model,
        )
        self.parser.add_instantiator(
            _InstantiatorFn(cli=self, key="data"),
            _get_module_type(self._datamodule_class),
            subclasses=self.subclass_mode_data,
        )
