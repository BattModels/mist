import smirk


def check_smile(smile):
    split = smirk.chemically_consistent_split(smile, is_smiles=True)
    assert isinstance(split, list)
    assert smile == "".join(split)
    return split

def check_selfie(selfie):
    split = smirk.chemically_consistent_split(selfie, is_smiles=False)
    assert isinstance(split, list)
    assert selfie == "".join(split)
    return split

def test_split_smiles():
    check_smile("COCCC(=O)N1CCN(C)C(C2=CN([C@@H](C)C3=CC=C(C(F)(F)F)C=C3)N=N2)C1")
    check_smile("CNC(C)Cc1ccccc1")

def test_split_selfies():
    check_selfie("[C][N][N][=C][Branch2][Ring2][C][C][=Branch1][C][=O][N][C][C][C][C][C][O][C][C][=C][N][Branch1][=N][C][=C][C][=C][C][Branch1][C][Br][=C][Ring1][#Branch1][F][N][=N][Ring1][=N][C][=C][Ring2][Ring1][N][O]")
    check_selfie("[C][C][Branch1][C][C][S][C][Branch1][C][C][C][=Branch1][C][=O][N][C][C][C][C][=C][C][Branch2][Ring1][C][C][=Branch1][C][=O][N][C][C][C][C][C][=Branch1][C][=O][N][C][Ring1][#Branch1][=C][C][=C][Ring1][P][C][Ring2][Ring1][=Branch1]")