import smirk


def test_smirk():
    tok = smirk.SmirkTokenizer("vocab.json")
    smile = "COCCC(=O)N1CCN(C)C(C2=CN([C@@H](C)C3=CC=C(C(F)(F)F)C=C3)N=N2)C1"
    emb = tok.encode(smile)
    assert len(emb.ids) == 62
    smile_out = tok.decode(emb.ids)
    print(smile_out)
    assert smile_out == smile


def test_encode_batch():
    tok = smirk.SmirkTokenizer("vocab.json")
    smile = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
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
