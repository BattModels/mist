import csv
import numpy as np
import pytest
import torch

from electrolyte_fm.data_modules import ComponentDataModule


@pytest.fixture(scope="module")
def tmp_dataset(tmp_path_factory):
    """Create a dummy CSV dataset with two targets."""
    root = tmp_path_factory.mktemp("mixture_data")

    def write_split(split: str, n_rows: int):
        fn = root / f"{split}.csv"
        with fn.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["smi1", "x1", "smi2", "x2", "temperature", "propA", "propB"]
            )
            rng = np.random.default_rng(0)
            for i in range(n_rows):
                writer.writerow(
                    [
                        "CCO",
                        rng.random(),
                        "CO",
                        rng.random(),
                        298.15,
                        rng.random(),  # propA
                        rng.random(),  # propB
                    ]
                )

    write_split("train", 4)
    write_split("val", 2)
    write_split("test", 2)
    return root


def test_multi_target_collation(tmp_dataset):
    """Check that target stacking & masks behave for multiple targets."""

    dm = ComponentDataModule(
        path=str(tmp_dataset),
        target_col=["propA", "propB"],
        batch_size=2,
        num_workers=0,
        include_temperature=True,
        tokenizer="smirk",
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    assert batch["target"].shape == (2, 2)  # (B, n_targets)
    assert batch["target_mask"].shape == (2, 2)
    assert batch["input_ids_0"].ndim == 2  # (B, seq_len)
    assert batch["input_ids_1"].shape[:1] == (2,)  # second component

    assert batch["target_mask"].sum() == 0

    assert "temperature" in batch
    assert batch["temperature"].shape == (2,)


def test_dataloader_iteration(tmp_dataset):
    """Ensure val-loader yields batches without error and correct keys."""

    dm = ComponentDataModule(
        path=str(tmp_dataset),
        target_col=["propA", "propB"],
        batch_size=2,
        num_workers=0,
        include_temperature=False,
        tokenizer="smirk",
    )
    dm.setup("validate")

    for split_loader in [
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
    ]:
        batch = next(iter(split_loader))
        required_keys = {
            "input_ids_0",
            "attention_mask_0",
            "composition_0",
            "input_ids_1",
            "attention_mask_1",
            "composition_1",
            "target",
            "target_mask",
        }
        assert required_keys.issubset(batch.keys())
        assert batch["target"].dtype == torch.float32
        assert batch["target_mask"].dtype == torch.bool
