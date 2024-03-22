#!/usr/bin/env python
# For Help: submit.py --help
# NOTE: Must activate environment (`poetry shell`) first
from __future__ import annotations
import jinja2
import typer
import os
import yaml
import json
from copy import deepcopy
from pathlib import Path
from typing import List
import rich
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax

cli = typer.Typer(rich_markup_mode="markdown")
console = rich.console.Console(stderr=True)


def parse_data(path) -> dict:
    with open(path, "r") as fid:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(fid)
        return json.load(fid)


class RelEnvironment(jinja2.Environment):
    """Override join_path() to enable relative template paths."""

    def join_path(self, template, parent):
        return str(Path(parent).parent.joinpath(template))


def merge_config(a: dict, b: dict):
    if not isinstance(b, dict):
        return b
    result = deepcopy(a)
    for k, v in b.items():
        if k.endswith(":"):
            result[k[:-1]] = b[k]
        elif k in result and isinstance(v, dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


@cli.command()
def compose(
    file: str = typer.Argument(
        file_okay=True,
        dir_okay=False,
        help="Jinja2 Template to render",
    ),
    data: List[str] = typer.Option(
        [],
        file_okay=True,
        dir_okay=False,
        help="YAML or JSON file specifying the template's variables. Multiple files can be provided and will be resolved sequentially",
    ),
    default: bool = typer.Option(
        True,
        help="Use the default.yaml file next to the template",
    ),
    script_config: bool = typer.Option(
        True,
        help="Apply the *.yaml file next to the template",
    ),
    confirm: bool = typer.Option(
        __name__ == "__main__",  # Default to asking for confirmation if interactive
        help="Display configuration and script befor printing",
    ),
):
    """
    Render a template `file` with using the values from `data`

    # Examples

    - Render a Template: `./submit/submit.py submit/polaris.j2`\n
    - Submit on Polaris: `./submit/submit.py submit/polaris.j2 | qsub`\n
    - Use a different config: `./submit/submit.py --data path/to/other/config.yaml submit/polaris.j2`\n
    - Stack multiple configs: `./submit/submit.py --data path/to/first.yaml --data path/to/other.yaml`

    """
    env = RelEnvironment(
        loader=jinja2.FileSystemLoader(os.getcwd()),
        lstrip_blocks=False,
    )
    template = env.get_template(file)

    # Load data
    if default:
        default_path = Path(file).parent.joinpath("default.yaml")
        config = parse_data(default_path)
    else:
        config = dict()

    # Load host specific config
    if script_config:
        script_config_file = Path(file).with_suffix(".yaml")
        if script_config_file.is_file():
            config = merge_config(config, parse_data(script_config_file))

    # Overlay data files
    for file in data:
        config = merge_config(config, parse_data(Path(file)))

    # Generate Script
    script = template.render(config)

    if not confirm:
        print(script)
    else:
        console.print(
            Panel(
                JSON(json.dumps(config)),
                title="Configuration",
            ),
            Panel(
                Syntax(script, "bash", background_color="default"),
                title="Script",
            ),
        )

        if Confirm.ask("Submit?", console=console):
            print(script)


if __name__ == "__main__":
    cli()
