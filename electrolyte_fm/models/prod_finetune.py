# Finetuned Models for Inference
import torch
import json
from pathlib import Path
from transformers import AutoModel, AutoConfig, DataCollatorWithPadding
from electrolyte_fm.models.prediction_task_head import PredictionTaskHead
from electrolyte_fm.models.normalize import AbstractNormalizer


def save_model(model, save_directory, safe_serialization=False):
    if safe_serialization:
        from safetensors.torch import save_model

        save_model(model, Path(save_directory, "model.safetensors"))
    else:
        torch.save(model.state_dict(), Path(save_directory, "model.pt"))


def load_model(model, save_directory):
    if (file := Path(save_directory, "model.safetensors")).is_file():
        from safetensors.torch import load_model

        load_model(model, file)
    elif (file := Path(save_directory, "model.pt")).is_file():
        model.load_state_dict(torch.load(file, weights_only=True))
    else:
        raise RuntimeError("No model found")


class MISTFinetuned(torch.nn.Module):
    def __init__(self, encoder, task_network, transform, channels=None):
        super().__init__()
        self.encoder = encoder
        self.task_network = task_network
        self.transform = transform
        self.channels = channels

    def forward(self, input):
        hs = self.encoder(input["input_ids"]).last_hidden_state
        y = self.task_network(hs)
        return self.transform.forward(y)

    def save_pretrained(self, save_directory, safe_serialization=False):
        config = {
            "encoder": self.encoder.config.to_diff_dict(),
            "task_network": {
                "embed_dim": self.encoder.config.hidden_size,
                "output_size": self.task_network.final.out_features,
                "dropout": self.task_network.dropout1.p,
            },
            "transform": self.transform.to_config(),
            "channels": self.channels,
        }

        Path(save_directory, "config.json").write_text(json.dumps(config, indent=4))
        save_model(self, save_directory, safe_serialization)

    def predict(self, smi: list[str], tokenizer):
        batch = tokenizer(smi)
        collate_fn = DataCollatorWithPadding(tokenizer)
        batch = collate_fn(batch).to(self.encoder.device)
        out = self(batch)
        if self.channels is None:
            return out
        return {k: out[:, idx].cpu().detach() for idx, k in enumerate(self.channels)}

    @classmethod
    def from_pretrained(self, save_directory: str):
        config = json.loads(Path(save_directory, "config.json").read_text())
        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        task_network = PredictionTaskHead(**config["task_network"])
        transform = AbstractNormalizer.get(
            config["transform"]["class"], config["transform"]["num_outputs"]
        )

        # Instantiate model
        model = MISTFinetuned(encoder, task_network, transform, config["channels"])
        load_model(model, save_directory)
        return model


class MISTMultiTask(torch.nn.Module):
    def __init__(self, encoder, task_networks, transforms, channels=None):
        super().__init__()
        self.encoder = encoder
        self.task_networks = torch.nn.ModuleList(task_networks)
        self.transforms = torch.nn.ModuleList(transforms)
        assert len(self.task_networks) == len(self.transforms)
        self.channels = channels

    def forward(self, input):
        hs = self.encoder(input["input_ids"]).last_hidden_state
        out = []
        for tn, tf in zip(self.task_networks, self.transforms):
            out.append(tf.forward(tn(hs)))

        return torch.cat(out, dim=-1)

    def predict(self, smi: list[str], tokenizer):
        batch = tokenizer(smi)
        collate_fn = DataCollatorWithPadding(tokenizer)
        batch = collate_fn(batch).to(self.encoder.device)
        out = self(batch).detach().cpu()
        if self.channels is None:
            return out
        return {k: out[:, idx] for idx, k in enumerate(self.channels)}

    def save_pretrained(self, save_directory, safe_serialization=False):
        config = {
            "encoder": self.encoder.config.to_diff_dict(),
            "task_networks": [
                {
                    "embed_dim": self.encoder.config.hidden_size,
                    "output_size": tn.final.out_features,
                    "transform": tf.__class__.__name__,
                    "dropout": tn.dropout1.p,
                }
                for tn, tf in zip(self.task_networks, self.transforms)
            ],
            "channels": self.channels,
        }
        Path(save_directory, "config.json").write_text(json.dumps(config, indent=4))
        save_model(self, save_directory, safe_serialization)

    @classmethod
    def from_pretrained(self, save_directory: str):
        config = json.loads(Path(save_directory, "config.json").read_text())
        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)

        task_networks = []
        transforms = []
        for tc in config["task_networks"]:
            transforms.append(
                AbstractNormalizer.get(tc.pop("transform"), tc["output_size"])
            )
            task_networks.append(PredictionTaskHead(**tc))

        model = MISTMultiTask(encoder, task_networks, transforms, config["channels"])
        load_model(model, save_directory)
        return model
