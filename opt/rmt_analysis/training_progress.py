import weightwatcher as ww
from electrolyte_fm.interpretibility.attention_maps import (
    maybe_load_model_and_tokenizer,
)
import glob
import pickle
import re


if __name__ == "__main__":
    tokenizer = None
    pattern = re.compile(r"step=(\d+)(?=\.ckpt)")
    records = {}
    checkpoint_paths = glob.glob(
        "/scratch/venkvis_root/venkvis/abhutani/mist-1.8B-dh61satt/checkpoints/weights-step=**.ckpt"
    )
    for checkpoint in checkpoint_paths:
        step = int(pattern.search(checkpoint).group(1))
        save_path = checkpoint.replace(".ckpt", ".csv")
        save_path = save_path.replace("checkpoints", "ww_analysis")
        checkpoint, tokenizer, tokenizer_name = maybe_load_model_and_tokenizer(
            checkpoint, tokenizer
        )
        watcher = ww.WeightWatcher(model=checkpoint)
        details = watcher.analyze(
            layers=[],
            min_evals=50,
        )
        describe = watcher.describe(layers=[], min_evals=50)
        records[step] = watcher.get_ESD()
        details.to_csv(save_path)
    with open("dh61satt_training_progress.pkl", "wb") as f:
        pickle.dump(records, f)
