"""
This is a standalone, importable SCScorer model. It does not have tensorflow as a
dependency and is a more attractive option for deployment. The calculations are
fast enough that there is no real reason to use GPUs (via tf) instead of CPUs (via np)

# Changelog
- Originally from https://github.com/Connor-Coley/scscore (37090a6aa8220408f572de20224c33c221312d16)
- Modified by Alexius Wadell to download model if not provided
"""

from pathlib import Path
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import json
import gzip
import six

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def download_model(path: Path, url: str = None) -> Path:
    from urllib.request import urlopen

    url = (
        url
        or "https://github.com/connorcoley/scscore/raw/refs/heads/master/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"
    )
    with urlopen(url) as fid:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as out:
            out.write(fid.read())
    return path


class SCScorer:
    def __init__(self, score_scale=score_scale):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False
        self._countlike = False
        self.FP_len = FP_len
        self.FP_rad = FP_rad

    def restore(self, weight_path=None, FP_rad=FP_rad, FP_len=FP_len):
        # Download default model if not provided
        if weight_path is None:
            weight_path = Path(
                Path(__file__).parent, "models", "full_reaxys_model_1024bool.json.gz"
            )
            if not weight_path.exists():
                download_model(weight_path)

        self.FP_len = FP_len
        self.FP_rad = FP_rad
        self._load_vars(weight_path)
        self._restored = True
        self._countlike = "uint8" in weight_path.name or "counts" in weight_path.name
        return self

    def mol_to_fp(self, mol):
        if self._countlike:
            if mol is None:
                return np.array((self.FP_len,), dtype=np.uint8)
            fp = AllChem.GetMorganFingerprint(
                mol, self.FP_rad, useChirality=True
            )  # uitnsparsevect
            fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
            for k, v in six.iteritems(fp.GetNonzeroElements()):
                fp_folded[k % self.FP_len] += v
            return np.array(fp_folded)

        else:
            if mol is None:
                return np.zeros((self.FP_len,), dtype=np.float32)
            return np.array(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.FP_rad, nBits=self.FP_len, useChirality=True
                ),
                dtype=bool,
            )

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError("Must restore model weights!")
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = i == len(self.vars) - 2
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi="", v=False):
        if not smi:
            return ("", 0.0)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v:
                print("Could not get fingerprint?")
            cur_score = 0.0
        else:
            # Run
            cur_score = self.apply(fp)[0]
            if v:
                print("Score: {}".format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ""
        return smi, cur_score

    def _load_vars(self, weight_path):
        if weight_path.name.endswith("pickle"):
            import cPickle as pickle

            with open(weight_path, "rb") as fid:
                self.vars = pickle.load(fid)
                self.vars = [x.tolist() for x in self.vars]
        elif weight_path.name.endswith("json.gz"):
            with gzip.GzipFile(weight_path, "r") as fin:  # 4. gzip
                json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)
                json_str = json_bytes.decode("utf-8")  # 2. string (i.e. JSON)
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]


if __name__ == "__main__":
    model = SCScorer()
    model.restore()
    smis = ["CCCOCCC", "CCCNc1ccccc1"]
    for smi in smis:
        smi, sco = model.get_score_from_smi(smi)
        print(f"{sco:.4f} <--- {smi}")
