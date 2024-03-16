import smirk
from pathlib import Path

VOCAB_FILE = Path(__file__).parent.parent.joinpath("vocab_selfies.json").resolve()
assert VOCAB_FILE.is_file()


def test_smirk():
    tok = smirk.SmirkTokenizer(str(VOCAB_FILE))
    smile = """[C][O][C][C][C][=Branch1][C][=O][N][C][C][N][Branch1][C][C][C][Branch2][Ring1]
               [P][C][=C][N][Branch2][Ring1][#Branch1][C@@H1][Branch1][C][C][C][=C][C][=C][Branch1]
               [=Branch2][C][Branch1][C][F][Branch1][C][F][F][C][=C][Ring1][#Branch2][N][=N][Ring1]
               [P][C][Ring2][Ring1][Branch2]"""
    emb = tok.encode(smile)
    assert len(emb.ids) == 62
    smile_out = tok.decode(emb.ids)
    print(smile_out)
    assert smile_out == smile


def test_encode_batch():
    tok = smirk.SmirkTokenizer(str(VOCAB_FILE))
    smile = [
        """[C][N][C][=N][C][=C][Ring1][Branch1][C][=Branch1][C][=O][N][Branch1][=Branch2][C]
        [=Branch1][C][=O][N][Ring1][Branch2][C][C]""",
        """[C][C][N][Branch1][Ring1][C][C][C][=Branch1][C][=O][C@H1][C][N][Branch2][Ring1]
        [Branch2][C@@H1][C][C][=C][NH1][C][=C][Ring1][Branch1][C][=Branch1][=Branch1][=C][C]
        [=C][Ring1][=Branch1][C][Ring1][N][=C][Ring1][S][C]""",
    ]
    emb = tok.encode_batch(smile)
    assert len(emb) == 2
    assert len(emb[0].ids) == 28
    assert len(emb[1].ids) == 53  # `@@` is one token
    assert len(emb[0].ids) == len(emb[0].attention_mask)
    assert len(emb[0].ids) == len(emb[0].type_ids)
    smile_out = tok.decode_batch([e.ids for e in emb])
    assert len(smile_out) == len(smile)
    assert smile_out == smile
    assert smile_out[0] == smile[0]
