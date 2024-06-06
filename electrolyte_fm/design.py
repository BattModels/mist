import json
from copy import deepcopy

import pyDOE2
import typer
import yaml
from rich.console import Console
from rich.table import Table

__doc__ = """
# Hyperparameter Sweeps for MIST

CLI tool for generating hyperparameter sweeps using design of experiment. All commands
take the form `python -m electrolye_fm.design CMD OPTS... sweep.yml`, where:

- **CMD**: Is the form of experiment design (See below for options)
- **OPTS...**: CMD specific flags (See subcommand docs for details)
- **sweep.yml**: A YAML file describing the parameters to sweep, mapping of swept parameters to configuration and job submission

## Example `sweep.yml`
```yaml
config:
  train:
    trainer.max_steps: $steps
    data.tokenizer: $tokenizer

command: python submit/submit.py submit/polaris.j2 ... $json --no-confirm | bash

space:
  steps:
   - 10_000
   - 20_000

  tokenizer:
    - smirk
    - ibm/MoLFormer-XL-both-10pct
```

> The above YAML doesn't render quite right (https://github.com/tiangolo/typer/issues/447).
> See the code to get the right spacing

The following describes the `config`, `command` and `space` keys

- **config**: Defines a YAML config identical to the input format of submit/submit.
    Parameters from `space` will be interpolated using `$varname`
- **command**: Defines the command that will be generated for each experiment.
  The contents of `config` will be interpolated into `$json` as a `--json` flag.
- **space**: Defines a sets of factors and discrete levels for each factor.

"""

# TODO: replace with `help=__doc__` when https://github.com/tiangolo/typer/issues/447 is fixed
cli = typer.Typer(rich_markup_mode="markdown", help=__doc__.replace("\n", "\n\n"))


class HyperSpace:
    def __init__(self, data: dict) -> None:
        self._data = data
        self._keys = list(self._data.keys())
        self._keys.sort()

    @property
    def n(self):
        return len(self._keys)

    @property
    def levels(self):
        return [len(self._data[k]) for k in self._keys]

    @property
    def factors(self):
        return self._keys

    def get_design(self, experiments: list[list[int]]) -> list[dict]:
        designs = []
        for experiment in experiments:
            designs.append(
                {k: self._data[k][level] for k, level in zip(self._keys, experiment)}
            )
        return designs

    @classmethod
    def from_fid(cls, fid):
        data = yaml.safe_load(fid)
        return HyperSpace(data)


class PairedHyperSpace(HyperSpace):
    def __init__(self, data: dict) -> None:
        super().__init__(data)
        self.base_axis = self._get_key(self._data)

    def _get_key(self, data: dict) -> str:
        return list(data.keys())[0]

    def _get_value(self, data: dict) -> str:
        return list(data.values())[0]

    @property
    def n(self):
        # only implemented for two levels
        return 2

    @property
    def base_axis_values(self) -> list[str]:
        return list(self._data[self.base_axis].keys())

    @property
    def spaces(self) -> list[HyperSpace]:
        hp_list = []
        for value in self.base_axis_values:
            space = {
                self.base_axis: [
                    value,
                ]
            }
            _nested = self._data[self.base_axis][value]
            space[self._get_key(_nested)] = self._get_value(_nested)
            hp_list.append(HyperSpace(space))
        return hp_list


def interpolate_design(config: dict, design: dict):
    """Recursively replace values in config with entries in design"""
    out = dict()
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("$"):
            out[k] = design[v[1:]]
        elif isinstance(v, dict):
            out[k] = interpolate_design(v, design)
        else:
            out[k] = v

    return out


def design_experiment(f, file: typer.FileText):
    data = yaml.safe_load(file)
    if "paired_space" in data.keys():
        if "space" in data.keys():
            raise NotImplementedError
        phs = PairedHyperSpace(data["paired_space"])
        hs_list = phs.spaces
    else:
        hs_list = [
            HyperSpace(data["space"]),
        ]
    target_config = deepcopy(data["config"])
    for hs in hs_list:
        for expr in hs.get_design(f(hs.levels)):
            config = interpolate_design(target_config, expr)
            config_json = json.dumps(config)
            cmd = data["command"].replace("$json", "--json '" + config_json + "'")
            print(cmd)


def display_experiment(f, file: typer.FileText):
    data = yaml.safe_load(file)
    if "paired_space" in data.keys():
        if "space" in data.keys():
            raise NotImplementedError
        phs = PairedHyperSpace(data["paired_space"])
        hs_list = phs.spaces
    else:
        hs_list = [
            HyperSpace(data["space"]),
        ]
    table = Table()
    columns = set()
    for hs in hs_list:
        for factor in hs.factors:
            if factor not in columns:
                table.add_column(str(factor))
                columns.add(str(factor))

        for expr in hs.get_design(f(hs.levels)):
            table.add_row(*[str(x) for x in expr.values()])

    console = Console(stderr=True)
    console.print(table)

    # Rewind the file for later commands
    file.seek(0)


@cli.command()
def gsd(
    file: typer.FileText,
    r: int = typer.Option(
        3,
        "--reduce",
        help="Factor by which to reduce the design space by",
        min=2,
    ),
    display: bool = typer.Option(
        False, "-d", "--display", help="Display the experiment on stderr"
    ),
):
    """
    Generalized Subset Design: Multi-factor design with varying levels per factor

    See: https://doi.org/10.1021/acs.analchem.7b00506
    """

    def f(levels):
        return pyDOE2.gsd(levels, r)

    if display:
        display_experiment(f, file)

    design_experiment(f, file)


@cli.command()
def fullfact(
    file: typer.FileText,
    display: bool = typer.Option(
        False, "-d", "--display", help="Display the experiment on stderr"
    ),
):
    """Full Factorial Design"""

    def f(levels):
        return pyDOE2.fullfact(levels).astype(int).tolist()

    if display:
        display_experiment(f, file)

    design_experiment(f, file)


if __name__ == "__main__":
    cli()
