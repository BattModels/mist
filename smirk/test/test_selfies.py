import pytest

from smirk import SmirkSelfiesFast


def test_wikipedia():
    tok = SmirkSelfiesFast()
    examples = [
        "[N][#N]",
        "[C][N][=C][=O]",
        "[Cu+2].[O-1][S][=Branch1][C][=O][=Branch1][C][=O][O-1]",
        "[O][=C][C][=C][C][=C][Branch1][C][O][C][Branch1][Ring1][O][C][=C][Ring1][=Branch2][C][O][C][=C][C][Branch1][Ring1][C][=O][=C][C][=C][Ring1][Branch2][O]",
        "[C][C][C][=Branch1][C][=C][C][=C][C][=N+1][Ring1][Branch1][C][=C][C][=C][Ring1][=Branch1][NH1][C][=C][Ring1][Branch1][C][=C][C][=C][Ring1][=Branch1][C][C][C][=C][N+1][=C][C][=C][C][=C][C][=C][C][=C][Ring1][=Branch1][NH1][C][Ring1][=Branch2][=C][Ring1][=N][C][=C][Ring1][P]",
        "[C][N][C][C][C][C@H1][Ring1][Branch1][C][=C][C][=C][N][=C][Ring1][=Branch1]",
        r"[C][C][C][C@@H1][Branch1][C][O][C][C][\\C][=C][\\C][=C][\\C][#C][C][#C][\\C][=C][\\C][O][C][C][C][C@@H1][Branch1][C][O][C][C][/C][=C][/C][=C][/C][#C][C][#C][/C][=C][/C][O]",
        r"[C][C][=C][Branch2][Ring2][Ring2][C][=Branch1][C][=O][C][C@@H1][Ring1][=Branch1][O][C][=Branch1][C][=O][C@@H1][C@H1][Branch1][Branch2][C][Ring1][Ring1][Branch1][C][C][C][/C][=C][Branch1][C][\\C][/C][=Branch1][C][=O][O][C][C][/C][=C][\\C][=C]",
        "[O][C][=C][C@H1][Branch1][Branch1][C@H1][Ring1][Branch1][O][C][=C][Ring1][Ring1][C][=C][Branch1][Ring1][O][C][C][=C][Ring1][Branch2][O][C][=Branch1][C][=O][C][=C][Ring1][#Branch1][C][C][C][=Branch1][C][=O][Ring1][Branch1]",
        "[O][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][Ring1][Branch2]",
        "[O][C][C@@H1][Branch1][C][O][C@@H1][Branch1][C][O][C@H1][Branch1][C][O][C@@H1][C@@H1][Ring1][#Branch1][C][=C][Branch1][C][O][C][Branch1][Ring1][O][C][=C][Branch1][C][O][C][=C][Ring1][#Branch2][C][=Branch1][C][=O][O][Ring1][#C]",
        r"[C][C][=Branch1][C][=O][O][C][C][C][Branch1][C][/C][=C][\\C][C@H1][Branch1][=Branch1][C][Branch1][C][C][=C][C][C][C][=C]",
        "[C][C][C@H1][Branch1][C][O][C][C][C@@][Ring1][Ring2][C][C][C][O][Ring1][Branch1]",
        "[C][C][Branch1][C][C][C@@][C][C@@H1][Ring1][Ring1][C@@H1][Branch1][C][C][C][=Branch1][C][=O][C][Ring1][Branch2]",
        "[O][C][C][C][=C][Branch1][C][C][N+1][=Branch1][Branch1][=C][S][Ring1][=Branch1][C][C][=C][N][=C][Branch1][C][C][N][=C][Ring1][#Branch1][N]",
    ]
    for example in examples:
        code = tok(example)
        assert tok.unk_token_id not in code["input_ids"]


def test_selfies():
    tok = SmirkSelfiesFast()
    assert tok.tokenize("[N][#N]") == ["N", "#", "N"]
    assert tok.tokenize("[O][=C][C][=C][C][=C][C][=C][Ring1][=Branch1]") == [
        "O",
        "=",
        "C",
        "C",
        "=",
        "C",
        "C",
        "=",
        "C",
        "C",
        "=",
        "C",
        "Ring",
        "1",
        "=",
        "Branch",
        "1",
    ]
    assert tok.tokenize("[Li][=C][C][S][=C][C][#S]") == [
        "Li",
        "=",
        "C",
        "C",
        "S",
        "=",
        "C",
        "C",
        "#",
        "S",
    ]
    assert tok.tokenize("[C][O][C]") == ["C", "O", "C"]
