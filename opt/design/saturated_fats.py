from argparse import ArgumentParser
import torch
import pandas as pd
from electrolyte_fm.models.prod_finetune import MISTFinetuned
from smirk import SmirkTokenizerFast

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=".")
    parser.add_argument("--output", default="sat_fats.csv")
    args = parser.parse_args()

    model = MISTFinetuned.from_pretrained(args.model).eval().to(device)
    tok = SmirkTokenizerFast()

    smi: list[str] = []
    for n in range(3, 40):
        # do nitrile
        smi.append("N#" + "C" * n)

    pred = model.predict(smi, tok)

    df = pd.DataFrame(pred, index=smi)
    df.to_csv(args.output, index_label="smi")
