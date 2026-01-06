import json
import torch
from pathlib import Path
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from .prod_finetune import load_model
from .normalize import AbstractNormalizer


class MISTIonicConductivity(torch.nn.Module):
    def __init__(self, encoder, task_network, tokenizer, n_components=38):
        super().__init__()
        self.encoder = encoder
        self.task_network = task_network
        self.tokenizer = tokenizer
        self.n_components = n_components

    def forward(self, batch, return_all=False):
        """
        Forward pass for mixture ionic conductivity prediction.

        Args:
            batch: Dictionary containing input_ids, attention_mask, composition,
                   and temperature for each component
            return_all: If True, return all parameters along with prediction

        Returns:
            Predicted conductivity and alpha (or all params if return_all=True)
        """
        mix_embedding = None
        for i in range(self.n_components):
            embedding = self.encoder(
                batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=True,
            ).last_hidden_state[:, 0, :]

            embedding = torch.stack(
                [
                    torch.mul(embedding[j, :], batch[f"composition_{i}"][j])
                    for j in range(embedding.shape[0])
                ]
            )
            if mix_embedding is None:
                mix_embedding = embedding
            else:
                mix_embedding += embedding

        params = self.task_network(mix_embedding, batch["temperature"])

        pred_unscaled = params["conductivity"]
        alpha = params["alpha"]
        beta = params["beta"]
        lmbda = params["beta"]

        exponent = torch.div(-1 * alpha + batch["composition_4"], lmbda)
        pred_decay = torch.mul((1 - beta), torch.exp(exponent)) + beta
        pred = torch.mul(pred_unscaled, pred_decay)

        # predicted conductivity*decay if salt molarity > alpha
        # else predicted conductivity
        pred = torch.where(batch["composition_4"] > alpha, pred, pred_unscaled)

        if return_all:
            return pred.view(-1, 1), params
        return pred.view(-1, 1), alpha

    def save_pretrained(self, save_directory, safe_serialization=False):
        """Save model configuration and weights."""
        config = {
            "architectures": [
                self.__class__.__name__,
            ],
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "encoder": self.encoder.config.to_diff_dict(),
            "task_network": {
                "embed_dim": self.encoder.config.hidden_size,
            },
            "n_components": self.n_components,
        }

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        Path(save_directory, "config.json").write_text(json.dumps(config, indent=4))

        # Save model state dict
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(self.state_dict(), Path(save_directory, "model.safetensors"))
        else:
            torch.save(self.state_dict(), Path(save_directory, "pytorch_model.bin"))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    def embed_mixture(
        self, smiles_list: list[list[str]], compositions: list[list[float]]
    ):
        """
        Generate embeddings for mixture components.

        Args:
            smiles_list: List of SMILES lists, where each inner list contains
                        SMILES for all components in a mixture
            compositions: List of composition lists corresponding to each mixture

        Returns:
            Mixture embeddings tensor
        """
        batch_size = len(smiles_list)
        mix_embeddings = []

        with torch.inference_mode():
            for batch_idx in range(batch_size):
                mix_embedding = None
                for comp_idx in range(
                    min(len(smiles_list[batch_idx]), self.n_components)
                ):
                    smi = smiles_list[batch_idx][comp_idx]
                    comp = compositions[batch_idx][comp_idx]

                    tokens = self.tokenizer([smi], return_tensors="pt", padding=True)
                    input_ids = tokens["input_ids"].to(self.encoder.device)
                    attention_mask = tokens["attention_mask"].to(self.encoder.device)

                    embedding = self.encoder(
                        input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=True,
                    ).last_hidden_state[:, 0, :]

                    embedding = embedding * comp

                    if mix_embedding is None:
                        mix_embedding = embedding
                    else:
                        mix_embedding += embedding

                mix_embeddings.append(mix_embedding)

        return torch.cat(mix_embeddings, dim=0).cpu()

    def predict(
        self,
        smiles_list: list[list[str]],
        compositions: list[list[float]],
        temperatures: list[float],
        salt_molarities: list[float],
        return_dict=True,
    ):
        """
        Predict ionic conductivity for mixtures.

        Args:
            smiles_list: List of SMILES lists for mixture components
            compositions: List of composition arrays for each mixture
            temperatures: List of temperatures
            salt_molarities: List of salt molarities (composition_4)
            return_dict: If True, return dictionary with detailed predictions

        Returns:
            Predictions (tensor or dict depending on return_dict)
        """
        batch_size = len(smiles_list)
        batch = {
            "temperature": torch.tensor(temperatures).float().to(self.encoder.device),
            "composition_4": torch.tensor(salt_molarities)
            .float()
            .to(self.encoder.device),
        }

        # Tokenize and prepare batch for each component
        for i in range(self.n_components):
            input_ids_list = []
            attention_mask_list = []
            comp_list = []

            for batch_idx in range(batch_size):
                if i < len(smiles_list[batch_idx]):
                    smi = smiles_list[batch_idx][i]
                    comp = compositions[batch_idx][i]
                else:
                    smi = "[H]"  # Dummy molecule
                    comp = 0.0

                tokens = self.tokenizer([smi], return_tensors="pt", padding=True)
                input_ids_list.append(tokens["input_ids"].squeeze(0))
                attention_mask_list.append(tokens["attention_mask"].squeeze(0))
                comp_list.append(comp)

            # Collate with padding
            collator = DataCollatorWithPadding(self.tokenizer)
            collated = collator(
                [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(input_ids_list, attention_mask_list)
                ]
            )

            batch[f"input_ids_{i}"] = collated["input_ids"].to(self.encoder.device)
            batch[f"attention_mask_{i}"] = collated["attention_mask"].to(
                self.encoder.device
            )
            batch[f"composition_{i}"] = (
                torch.tensor(comp_list).float().to(self.encoder.device)
            )

        with torch.inference_mode():
            pred, params = self(**batch, return_all=return_dict)

        if not return_dict:
            return pred.cpu()

        result = {
            "conductivity": pred.cpu(),
            "alpha": params["alpha"].cpu(),
            "beta": params["beta"].cpu(),
            "conductivity_unscaled": params["conductivity"].cpu(),
        }
        return result

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load model from saved directory."""
        config = json.loads(Path(save_directory, "config.json").read_text())

        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=True)

        from .physics_task_heads import VFTDecayTaskHead

        task_network = VFTDecayTaskHead(embed_dim=config["task_network"]["embed_dim"])

        tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=True)
        n_components = config.get("n_components", 5)

        model = cls(encoder, task_network, tokenizer, n_components)
        load_model(model, save_directory)
        return model


class MISTExcessPhysics(torch.nn.Module):
    def __init__(
        self,
        encoder,
        task_network,
        transform,
        tokenizer,
        n_components=2,
        temperature_normalization=(273, 400),
    ):
        super().__init__()
        self.encoder = encoder
        self.task_network = task_network
        self.transform = transform
        self.tokenizer = tokenizer
        self.n_components = n_components
        self.temperature_normalization = temperature_normalization

    def forward(self, batch, transform=True):
        """
        Forward pass for mixture property prediction.

        Args:
            batch: Dictionary containing input_ids, attention_mask for each component,
                   temperature, and composition data
            transform: If True, apply normalization transform to predictions

        Returns:
            Predicted property values
        """
        mn, mx = self.temperature_normalization
        # Normalize temperature once per mixture
        temperature = (batch["temperature"] - mn) / (mx - mn)  # (B,)
        batch["temperature"] = temperature

        for i in range(self.n_components):
            enc_out = self.encoder(
                input_ids=batch[f"input_ids_{i}"],
                attention_mask=batch[f"attention_mask_{i}"],
                return_dict=True,
                output_hidden_states=False,
            )

            token_seq = enc_out.last_hidden_state  # (B, L_i, d)
            padmask = batch[f"attention_mask_{i}"] == 0  # (B, L_i)  bool

            # Save for cross-attention fusion
            batch[f"tokens_{i}"] = token_seq.float()
            batch[f"padmask_{i}"] = padmask

            # Mean-pool tokens_i: single-molecule embedding
            pooled = token_seq.masked_fill(padmask.unsqueeze(-1), 0).mean(dim=1)
            batch[f"embedding_{i}"] = pooled

        # Property prediction
        pred = self.task_network(batch)  # (B, 1)

        if transform:
            pred = self.transform.forward(pred)  # Rescale to original units
        return pred

    def save_pretrained(self, save_directory, safe_serialization=False):
        """Save model configuration and weights."""
        config = {
            "architectures": [
                self.__class__.__name__,
            ],
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "encoder": self.encoder.config.to_diff_dict(),
            "task_network": {
                "embed_dim": self.encoder.config.hidden_size,
                "polynomial_order": getattr(self.task_network, "polynomial_order", 4),
                "n_components": self.n_components,
                "num_heads": getattr(self.task_network, "num_heads", 4),
                "include_linear_mixing": getattr(
                    self.task_network, "include_linear_mixing", True
                ),
                "fusion": getattr(self.task_network, "fusion", "attention"),
                "basis": self.task_network.__class__.__name__,
            },
            "transform": self.transform.to_config(),
            "n_components": self.n_components,
            "temperature_normalization": self.temperature_normalization,
        }

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        Path(save_directory, "config.json").write_text(json.dumps(config, indent=4))

        # Save model state dict
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(self.state_dict(), Path(save_directory, "model.safetensors"))
        else:
            torch.save(self.state_dict(), Path(save_directory, "pytorch_model.bin"))

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    def embed_components(
        self, smiles_list: list[list[str]], compositions: list[list[float]]
    ):
        """
        Generate embeddings for mixture components.

        Args:
            smiles_list: List of SMILES lists, where each inner list contains
                        SMILES for all components in a mixture
            compositions: List of composition lists corresponding to each mixture

        Returns:
            Dictionary with embeddings for each component
        """
        batch_size = len(smiles_list)
        component_embeddings = {i: [] for i in range(self.n_components)}

        with torch.inference_mode():
            for batch_idx in range(batch_size):
                for comp_idx in range(self.n_components):
                    if comp_idx < len(smiles_list[batch_idx]):
                        smi = smiles_list[batch_idx][comp_idx]
                    else:
                        smi = "[H]"  # Dummy molecule

                    tokens = self.tokenizer([smi], return_tensors="pt", padding=True)
                    input_ids = tokens["input_ids"].to(self.encoder.device)
                    attention_mask = tokens["attention_mask"].to(self.encoder.device)

                    enc_out = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=False,
                    )

                    token_seq = enc_out.last_hidden_state
                    padmask = attention_mask == 0
                    pooled = token_seq.masked_fill(padmask.unsqueeze(-1), 0).mean(dim=1)

                    component_embeddings[comp_idx].append(pooled)

        # Concatenate all embeddings per component
        result = {}
        for comp_idx in range(self.n_components):
            result[f"component_{comp_idx}"] = torch.cat(
                component_embeddings[comp_idx], dim=0
            ).cpu()

        return result

    def predict(
        self,
        smiles_list: list[list[str]],
        compositions: list[list[float]],
        temperatures: list[float],
        return_dict=False,
    ):
        """
        Predict mixture properties.

        Args:
            smiles_list: List of SMILES lists for mixture components
                        e.g., [["CCO", "CC"], ["CCCO", "CCC"]] for 2 binary mixtures
            compositions: List of composition arrays for each mixture
                         e.g., [[0.5, 0.5], [0.3, 0.7]]
            temperatures: List of temperatures (in Kelvin)
            return_dict: If True, return dictionary with detailed information

        Returns:
            Predictions (tensor or dict depending on return_dict)
        """
        batch_size = len(smiles_list)

        # Validate inputs
        assert len(compositions) == batch_size, "Mismatch in batch sizes"
        assert len(temperatures) == batch_size, "Mismatch in batch sizes"

        for i, (smiles, comps) in enumerate(zip(smiles_list, compositions)):
            assert len(smiles) == len(
                comps
            ), f"Mixture {i}: SMILES and composition lengths don't match"
            assert (
                len(smiles) <= self.n_components
            ), f"Mixture {i}: Too many components (max {self.n_components})"

        batch = {
            "temperature": torch.tensor(temperatures, dtype=torch.float32).to(
                self.encoder.device
            )
        }

        # Tokenize and prepare batch for each component
        for i in range(self.n_components):
            input_ids_list = []
            attention_mask_list = []
            comp_list = []

            for batch_idx in range(batch_size):
                if i < len(smiles_list[batch_idx]):
                    smi = smiles_list[batch_idx][i]
                    comp = compositions[batch_idx][i]
                else:
                    smi = "[H]"  # Dummy molecule for padding
                    comp = 0.0

                tokens = self.tokenizer([smi], return_tensors="pt", padding=True)
                input_ids_list.append(tokens["input_ids"].squeeze(0))
                attention_mask_list.append(tokens["attention_mask"].squeeze(0))
                comp_list.append(comp)

            # Collate with padding
            collator = DataCollatorWithPadding(self.tokenizer)
            collated = collator(
                [
                    {"input_ids": ids, "attention_mask": mask}
                    for ids, mask in zip(input_ids_list, attention_mask_list)
                ]
            )

            batch[f"input_ids_{i}"] = collated["input_ids"].to(self.encoder.device)
            batch[f"attention_mask_{i}"] = collated["attention_mask"].to(
                self.encoder.device
            )
            batch[f"composition_{i}"] = torch.tensor(comp_list, dtype=torch.float32).to(
                self.encoder.device
            )

        with torch.inference_mode():
            pred = self(batch, transform=True)

        if not return_dict:
            return pred.cpu()

        result = {
            "prediction": pred.cpu(),
            "smiles": smiles_list,
            "compositions": compositions,
            "temperatures": temperatures,
        }
        return result

    def predict_single(
        self, smiles: list[str], composition: list[float], temperature: float
    ):
        """
        Convenience method to predict for a single mixture.

        Args:
            smiles: List of SMILES for the mixture components
            composition: List of mole fractions (should sum to 1.0)
            temperature: Temperature in Kelvin

        Returns:
            Predicted property value (scalar tensor)
        """
        pred = self.predict([smiles], [composition], [temperature])
        return pred.squeeze()

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load model from saved directory."""
        config = json.loads(Path(save_directory, "config.json").read_text())

        encoder_config = AutoConfig.for_model(
            config["encoder"]["model_type"]
        ).from_dict(config["encoder"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)

        from .polynomial_task_head import PolynomialHead

        basis_name = config["task_network"]["basis"]
        task_network_config = {
            "embed_dim": config["task_network"]["embed_dim"],
            "polynomial_order": config["task_network"]["polynomial_order"],
            "n_components": config["task_network"]["n_components"],
            "num_heads": config["task_network"]["num_heads"],
            "include_linear_mixing": config["task_network"]["include_linear_mixing"],
            "fusion": config["task_network"]["fusion"],
        }

        # Instantiate the appropriate polynomial head
        task_network = PolynomialHead.get_class(basis_name)(**task_network_config)

        transform = AbstractNormalizer.get(
            config["transform"]["class"], config["transform"]["num_outputs"]
        )

        tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=True)
        n_components = config.get("n_components", 2)
        temperature_normalization = tuple(
            config.get("temperature_normalization", (273, 400))
        )

        model = cls(
            encoder,
            task_network,
            transform,
            tokenizer,
            n_components,
            temperature_normalization,
        )

        model = cls(encoder, task_network, tokenizer, n_components)
        load_model(model, save_directory)
        return model
