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
    location,
    dataset,
    tokenizer="smirk",
    encoding="smiles-kekule",
    batch_size,
    hidden_size,
    n_layers,
)

{
  nodes: 1,
  gpus_per_node: 1,
  container: '/lustre/fs0/awadell/sqsh-files/mist+pytorch+25.01+v2.sif',
  env: {
    JOBID: '$SLURM_JOB_ID',
    PMIX_MCA_gds: 'hash',
    NCCL_TOPO_FILE: '/cm/shared/etc/ndv4-topo.xml',
    MELLANOC_VISIBLE_DEVICES: 'all',
  },
  program: "-m electrolyte_fm.models.linear_probe",
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
        class_path: 'electrolyte_fm.models.linear_probe.per_layer_probe',
        init_args: {
            hidden_size: std.parseInt(hidden_size),
            features: 5,
            location: location,
            n_layers: std.parseInt(n_layers),
        },
        },
    },
    data: {
        class_path: 'electrolyte_fm.data_modules.lipinski_dataset.LipinskiDataModule',
        init_args: {
        name_or_path: dataset,
        tokenizer: tokenizer,
        encoding: encoding,
        num_workers: 16,
        batch_size: std.parseInt(batch_size),
        },
    },
  },
}
"""

template = "submit/dgx.j2"


def submit(config: dict):
    script = render(template, config)
    print(script)
    subprocess.run("sbatch", input=script, text=True)


# Pretrained Models
datasets = ["tox21", "toxcast", "hiv"]
locations = ["output", "intermediate", "output.dense"]
models = [
    {
        "encoder_path": "ibm/MoLFormer-XL-both-10pct",
        "encoding": "smiles-canonical",
        "tokenizer": "ibm/MoLFormer-XL-both-10pct",
    },
    {
        "encoder_path": "./models/mist-ti624ev1-moleculenet/pretrained",
    },
    {
        "encoder_path": "./models/mist-1.8B-dh61satti",
        "batch_size": 16,
    },
]
models.extend(
    [
        {
            "encoder_class": "electrolyte_fm.models.prod_finetune.MISTFinetuned.from_pretrained",
            "encoder_path": f"./models/mist-ti624ev1-moleculenet/{dataset}",
        }
        for dataset in ["bace", "qm9", "bbbp", "muv", "qm8", "tmQM"]
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
    run.setdefault("encoder_class", "__main__.mlm_from_pretrained")
    run.setdefault("batch_size", 64)
    if run["encoder_class"] == "__main__.mlm_from_pretrained":
        model_config = AutoConfig.from_pretrained(
            run["encoder_path"],
            trust_remote_code=True,
        )
    else:
        model_config = get_mist_finetune_config(run["encoder_path"])

    run.setdefault("n_layers", model_config.num_hidden_layers)

    for c in dict_product({"location": locations, "dataset": datasets}):
        run.update(c)
        if run["location"] == "intermediate":
            hidden_size = model_config.intermediate_size
        else:
            hidden_size = model_config.hidden_size

        run["hidden_size"] = hidden_size
        run_config = jsonnet.evaluate_snippet(
            "snippet",
            config,
            tla_vars={k: str(v) for k, v in run.items()},
        )
        run_config = json.loads(run_config)
        submit(run_config)
