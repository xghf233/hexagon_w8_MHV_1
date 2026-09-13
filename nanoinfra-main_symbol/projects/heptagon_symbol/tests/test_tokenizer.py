import numpy as np
import pytest
import torch

from projects.heptagon_symbol import tokenizer as t

WORD = [0, 0, 0, 0, 0, 7]


def test_real_source_example_and_shift():
    seq = t.assemble_sequence(WORD, -48)
    assert seq.tokens.tolist() == [42, 0, 0, 0, 0, 0, 7, 43, 47, 96, 44, 45, 45, 45, 45, 45]
    assert seq.token_types.tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 1, 1, 1, 1]
    assert seq.loss_weights.tolist() == [0] * 8 + [1] * 3 + [0] * 5
    batch = t.next_token_batch(t.assemble_batch(np.array([WORD]), np.array([-48])))
    assert batch['idx'].shape == (1, 15)
    assert batch['targets'][0].tolist() == [-1] * 7 + [47, 96, 44] + [-1] * 5
    assert batch['token_types'][0, 7].item() == 1  # conditioning COEFF
    assert batch['target_types'][0, 7].item() == 2  # predicted sign
    assert batch['target_types'][0, 9].item() == 1  # predicted EOS
    assert batch['loss_weights'].sum().item() == 3
    assert t.decode_sequence(seq.tokens) == (('a11',) * 5 + ('a21',), -48)


def test_alphabet_layout_and_prompt():
    labels = ['a11', 'a17', 'a21', 'a37', 'a61', 'a67']
    ids = [0, 6, 7, 20, 35, 41]
    assert t.encode_word(labels) == ids
    assert t.decode_word(np.array(ids, dtype=np.uint8)) == tuple(labels)
    assert t.encode_prompt(ids) == [42, *ids, 43]
    assert len(t.encode_prompt(ids)) == t.PROMPT_LENGTH == 8
    assert (t.WORD_START, t.WORD_END) == (1, 7)
    layout = t.build_layout()
    assert (layout.vocab_size, layout.n_token_types) == (1048, 3)
    assert layout.classify_token_types(torch.arange(1048)).tolist() == [t.token_type_of(i) for i in range(1048)]
    assert t.encoding_metadata()['NUM_OFFSET'] == 48


@pytest.mark.parametrize('value', [0, 1, -1, -48, 48, 999, -999, 1000, -1000, 12334, 10**30])
def test_general_integer_roundtrip(value):
    tokens = t.encode_coefficient(value)
    assert t.decode_coefficient(tokens) == value
    assert t.decode_generated(tokens + [t.EOS_ID, t.PAD_ID]) == value
    seq = t.assemble_sequence(WORD, value, sequence_len=32)
    assert t.decode_sequence(seq.tokens)[1] == value


def test_multiblock_and_unsigned_source_ids():
    assert t.encode_coefficient(-12334) == [47, 60, 382]
    words = np.array([WORD, [41] * 6], dtype=np.uint8)
    batch = t.assemble_batch(words, np.array([-48, 48], dtype=np.int16))
    for i, coeff in enumerate([-48, 48]):
        scalar = t.assemble_sequence(words[i], coeff)
        assert torch.equal(batch.tokens[i], scalar.tokens)
        assert torch.equal(batch.token_types[i], scalar.token_types)
        assert torch.equal(batch.loss_weights[i], scalar.loss_weights)
    assert t.assemble_sequence(WORD, 1000).loss_weights.sum().item() == 4


@pytest.mark.parametrize('word', [[0] * 5, [0] * 7, [-1] * 6, [42] * 6, [1.0] * 6, [True] * 6, 'a11a11'])
def test_invalid_word_ids(word):
    with pytest.raises(ValueError):
        t.encode_prompt(word)


@pytest.mark.parametrize('labels', ['a11a11a11a11a11a21', ['a00'] * 6, ['a11'] * 5])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        t.encode_word(labels)


@pytest.mark.parametrize('tokens', [[], [46], [48, 49], [46, 47], [46, 48, 49], [47, 48], [46, 1048], [True, 49], [46, 49.0], [46, 49, 44]])
def test_invalid_coefficients(tokens):
    with pytest.raises(ValueError):
        t.decode_coefficient(tokens)


@pytest.mark.parametrize('tokens', [[46, 49], [46, 49, 45], [46, 49, 44, 49], [46, 49, 44, 44]])
def test_generation_requires_eos_and_no_trailing_answer(tokens):
    with pytest.raises(ValueError):
        t.decode_generated(tokens)


def test_reject_truncation_and_invalid_numbers():
    for value in (True, 1.5, '48'):
        with pytest.raises(ValueError):
            t.encode_coefficient(value)
    for value in (-1, 1000, True):
        with pytest.raises(ValueError):
            t.num_id(value)
    for value in (-1, 1048, 1.5):
        with pytest.raises(ValueError):
            t.token_type_of(value)
    with pytest.raises(ValueError):
        t.assemble_sequence(WORD, 1000, sequence_len=11)
    with pytest.raises(ValueError):
        t.assemble_batch([WORD], [1000])
    with pytest.raises(ValueError):
        t.assemble_batch([WORD], [1.5])
    with pytest.raises(ValueError):
        t.assemble_batch([[0.0] * 6], [1])
    with pytest.raises(ValueError):
        t.assemble_batch([WORD], np.array([2**64 - 1], dtype=np.uint64))
    with pytest.raises(ValueError):
        t.assemble_batch([WORD], [[1]])


def test_strict_full_sequence_and_prompt_independence():
    positive = t.assemble_sequence(WORD, 48).tokens.tolist()
    negative = t.assemble_sequence(WORD, -48).tokens.tolist()
    assert positive[:8] == negative[:8] == t.encode_prompt(WORD)
    for position, replacement in [(0, 45), (7, 45), (1, 42), (10, 45), (15, 49)]:
        invalid = positive.copy()
        invalid[position] = replacement
        with pytest.raises(ValueError):
            t.decode_sequence(invalid)
