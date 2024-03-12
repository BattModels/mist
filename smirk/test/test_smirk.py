import smirk


def check_smile(smile):
    split = smirk.chemically_consistent_split(smile)
    assert isinstance(split, list)
    assert smile == "".join(split)
    return split


def test_split():
    check_smile("COCCC(=O)N1CCN(C)C(C2=CN([C@@H](C)C3=CC=C(C(F)(F)F)C=C3)N=N2)C1")
    check_smile("CNC(C)Cc1ccccc1")
