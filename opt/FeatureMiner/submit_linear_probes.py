from pathlib import Path
from copy import deepcopy
import json
import _jsonnet as jsonnet
import subprocess
from submit.submit import render
from submit.utils import dict_product
from transformers import AutoConfig

# jsonnet
config = """
function(
    encoder_class,
    encoder_path,
    tokenizer="smirk",
    encoding="smiles-kekule",
    batch_size,
    hidden_size,
    intermediate_size,
    n_layers,
)

{
  nodes: 1,
  gpus_per_node: 1,
  container: '/nfs/turbo/coe-venkvis/mist/mist+pytorch+25.01+v4.sif',
    walltime: "1:0:0",
  env: {
    JOBID: '$SLURM_JOB_ID',
    TORCH_EXTENSIONS_DIR: '${PWD}/.cache/torch_extensions',
    HF_HOME: '${PWD}/.cache/huggingface',
    TOKENIZERS_PARALLELISM: true,
  },
  program: "-m electrolyte_fm.models.linear_probe",
  stage: null,
  train: {
    trainer: {
        max_epochs: 1000, 
    },
    model: {
        model: {
        class_path: encoder_class,
        init_args: {
            name_or_path: encoder_path,
        },
        },
        probes: {
        class_path: 'electrolyte_fm.models.linear_probe.probe_everything',
        init_args: {
            hidden_size: std.parseInt(hidden_size),
            intermediate_size: std.parseInt(intermediate_size),
            features: 5,
            n_layers: std.parseInt(n_layers),
        },
        },
    },
    data: {
        class_path: 'electrolyte_fm.data_modules.lipinski_dataset.LipinskiDataModule',
        init_args: {
            path: "./lipinski",
            tokenizer: tokenizer,
            encoding: encoding,
            num_workers: 16,
            batch_size: std.parseInt(batch_size),
            randomize: true,
        },
    },
  },
}
"""

template = "submit/artemis.j2"


def submit(config: dict):
    script = render(template, config)
    print(script)
    exit()
    subprocess.run("sbatch", input=script, text=True)


# Pretrained Models
models = [
    {
        "encoder_path": "./models/mist-ti624ev1-moleculenet/pretrained",
    }
]
models.extend(
    [
        {
            "encoder_class": "__main__.encoder_from_finetuned",
            "encoder_path": f"./models/mist-ti624ev1-moleculenet/{dataset}",
        }
        for dataset in [
            "bace",
            "qm9",
            "bbbp",
            "muv",
            "qm8",
            "tmQM",
            "clintox",
        ]
    ]
)


def get_mist_finetune_config(path):
    config = json.loads(Path(path, "config.json").read_text())
    return AutoConfig.for_model(config["encoder"]["model_type"]).from_dict(
        config["encoder"]
    )


runs = []
for model in models:
    run = deepcopy(model)
    default_encoder = "electrolyte_fm.utils.cli.mlm_from_pretrained"
    run.setdefault("encoder_class", default_encoder)
    run.setdefault("batch_size", 128)
    if run["encoder_class"] == default_encoder:
        model_config = AutoConfig.from_pretrained(
            run["encoder_path"],
            trust_remote_code=True,
        )
    else:
        model_config = get_mist_finetune_config(run["encoder_path"])

    run["n_layers"] = model_config.num_hidden_layers
    run["hidden_size"] = model_config.hidden_size
    run["intermediate_size"] = model_config.intermediate_size
    run_config = jsonnet.evaluate_snippet(
        "snippet",
        config,
        tla_vars={k: str(v) for k, v in run.items()},
    )
    run_config = json.loads(run_config)
    submit(run_config)
