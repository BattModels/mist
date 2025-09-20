# requires RDKit >= 2022.09 (for DATIVE bonds)
from rdkit import Chem
from rdkit.Chem import AllChem
import math

ARENE_SMILES = {
    "benzene": "c1ccccc1",
    "naphthalene": "c1c2ccccc2ccc1",
    "pyrene": "c1cc2cccc3c2c4c1cccc4cc3",
    "coronene": "c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67",
}


def _mol_with_coords(smi: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smi)
    assert m is not None, smi
    return _embed_mol(m)


def _embed_mol(m: Chem.Mol):
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(m, maxIters=200)
    m = Chem.RemoveHs(m)
    return m


def _ring_centroid(mol: Chem.Mol, ring: tuple[int]) -> tuple[float, float, float]:
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in ring]
    ys = [conf.GetAtomPosition(i).y for i in ring]
    zs = [conf.GetAtomPosition(i).z for i in ring]
    n = float(len(ring))
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def _mol_centroid(mol: Chem.Mol) -> tuple[float, float, float]:
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    zs = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
    n = float(mol.GetNumAtoms())
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _pick_ring(mol: Chem.Mol, which: str) -> tuple[int, ...]:
    """Pick a 6-membered ring; 'inner' is closest to overall centroid; 'outer' farthest."""
    sssr = [tuple(r) for r in Chem.GetSymmSSSR(mol)]
    sixers = [r for r in sssr if len(r) == 6]
    if not sixers:
        raise ValueError("No 6-membered rings found in arene.")
    mcent = _mol_centroid(mol)
    scored = [(r, _dist(_ring_centroid(mol, r), mcent)) for r in sixers]
    scored.sort(key=lambda x: x[1])
    if which == "inner":
        return scored[0][0]
    elif which == "outer":
        return scored[-1][0]
    else:
        # default: inner for benzene/naphthalene/pyrene/coronene
        return scored[0][0]


def _adjacent_pairs(ring: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


def generate_metal_arene_complex(
    arene: str = "benzene",
    metal: str = "Fe",
    charge: int = 0,
    configuration: str = "auto",  # "auto" | "inner" | "outer"
    binding: str = "auto",  # "auto" | "eta2" | "eta6" | "center2"
    name: str | None = None,
) -> Chem.Mol:
    """
    Build an approximate metal–arene complex suitable for SMILES export.

    arene: one of {'benzene','naphthalene','pyrene','coronene'}
    metal: 'Ti','Cr','Fe','Ni', etc.
    charge: overall formal charge for the complex (0, +1, +2)
    configuration: which ring to bind on multi-ring arenes ('inner'|'outer'|'auto')
    binding:
        - 'eta2': two dative bonds to adjacent ring carbons (edge-binding)
        - 'eta6': dative bonds to all six ring carbons (haptic-like)
        - 'center2': two single bonds to opposite carbons (crude Ti-like)
        - 'auto': Ti→center2, else→eta2
        - 'auto6': Ti→eta6, else→eta2
    """
    if arene not in ARENE_SMILES:
        raise ValueError(f"Unknown arene '{arene}'")
    base = _mol_with_coords(ARENE_SMILES[arene])

    # choose ring
    ring_choice = "inner" if configuration == "auto" else configuration
    ring = _pick_ring(base, ring_choice)

    # decide binding mode
    if binding == "auto":
        if metal.capitalize() == "Ti":
            binding_mode = "center2"
        else:
            binding_mode = "eta2"

    elif binding == "auto6":
        binding_mode = "eta6" if metal.capitalize() == "Ti" else "eta2"

    else:
        binding_mode = binding

    # build editable mol and add metal atom
    em = Chem.EditableMol(base)
    pt = Chem.GetPeriodicTable()
    z = pt.GetAtomicNumber(metal)
    if z == 0:
        raise ValueError(f"Unknown metal symbol '{metal}'")
    metal_idx = em.AddAtom(Chem.Atom(z))
    # set metal charge on the final mol (RDKit wants it post-construction)
    mol = em.GetMol()

    # connect
    if binding_mode == "eta6":
        for ai in ring:
            em.AddBond(metal_idx, ai, Chem.BondType.DATIVE)
    elif binding_mode == "eta2":
        # pick the adjacent pair that is most "edge-like": farthest from ring centroid
        rc = _ring_centroid(mol, ring)
        pairs = _adjacent_pairs(ring)

        def pair_score(p):
            pos = mol.GetConformer()
            a = pos.GetAtomPosition(p[0])
            b = pos.GetAtomPosition(p[1])
            mid = ((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
            return _dist(mid, rc)

        a, b = max(pairs, key=pair_score)
        em.AddBond(metal_idx, a, Chem.BondType.DATIVE)
        em.AddBond(metal_idx, b, Chem.BondType.DATIVE)
    elif binding_mode == "center2":
        # connect to roughly opposite carbons (para-like) on the chosen ring
        # pick an atom and the atom ~3 steps away in the ring list
        a = ring[0]
        b = ring[3]
        em.AddBond(metal_idx, a, Chem.BondType.SINGLE)
        em.AddBond(metal_idx, b, Chem.BondType.SINGLE)
    else:
        raise ValueError(f"Unknown binding mode '{binding_mode}'")

    # set overall charge by putting it on the metal (simplest approximation)
    mol = Chem.RWMol(em.GetMol())
    mol.GetAtomWithIdx(metal_idx).SetFormalCharge(charge)
    mol = mol.GetMol()

    mol.RemoveAllConformers()

    # sanitize gently: allow dative bonds to remain
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
    )

    # add a name
    label = name or f"{metal}-on-{arene}-{binding_mode}-{ring_choice}-q{charge}"
    mol.SetProp("_Name", label)
    return mol


def smiles(mol: Chem.Mol) -> str:
    # Convert to SMILES
    p = Chem.rdmolfiles.SmilesWriteParams()
    p.canonical = True
    p.allHsExplicit = False
    p.allBondsExplicit = False
    p.doKekule = False
    p.includeDativeBonds = False
    return Chem.MolToSmiles(mol, p)


# --- quick usage examples ---
if __name__ == "__main__":
    # Ti on benzene, center-like two bonds (neutral)
    m1 = generate_metal_arene_complex(
        "benzene", "Ti", charge=0, configuration="inner", binding="auto"
    )
    print(m1.GetProp("_Name"), smiles(m1))

    # Fe on pyrene, outer ring, η2 dative bonds, 1+ cation
    m2 = generate_metal_arene_complex(
        "pyrene", "Fe", charge=1, configuration="outer", binding="eta2"
    )
    print(m2.GetProp("_Name"), smiles(m2))

    # Ni on coronene, outer ring, η6 representation (six dative links), neutral
    m3 = generate_metal_arene_complex(
        "coronene", "Ni", charge=0, configuration="outer", binding="eta6"
    )
    print(m3.GetProp("_Name"), smiles(m3))
