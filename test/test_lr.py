from electrolyte_fm.utils.lr_schedule import _get_cosine_relative_decay_with_warmup


def test_cosine_rel():
    assert 0 == _get_cosine_relative_decay_with_warmup(
        0,
        num_training_steps=100,
        num_warmup_steps=10,
        rel_decay=0.25,
    )
    assert 1.0 == _get_cosine_relative_decay_with_warmup(
        10,
        num_training_steps=100,
        num_warmup_steps=10,
        rel_decay=0.25,
    )
    assert 0.25 == _get_cosine_relative_decay_with_warmup(
        100,
        num_training_steps=100,
        num_warmup_steps=10,
        rel_decay=0.25,
    )
    assert (0.25 + 0.75 / 2) == _get_cosine_relative_decay_with_warmup(
        55,
        num_training_steps=100,
        num_warmup_steps=10,
        rel_decay=0.25,
    )
