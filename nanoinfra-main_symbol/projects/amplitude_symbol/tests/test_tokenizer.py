"""Tests for SymbolTokenizer — encode/decode, round-trip, padding, loss mask."""

import torch
import pytest

# Ensure nanoinfra root is importable
import os, sys
_NANOINFRA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _NANOINFRA_ROOT not in sys.path:
    sys.path.insert(0, _NANOINFRA_ROOT)

from projects.amplitude_symbol.tokenizer import (
    BOS_ID,
    COEFF_ID,
    COEFF_OFFSET,
    CONTROL_OFFSET,
    CONTROL_SIZE,
    COEFF_SIZE,
    EOS_ID,
    IGNORE_INDEX,
    MINUS_ID,
    PAD_ID,
    PLUS_ID,
    VOCAB_SIZE,
    WORD_OFFSET,
    num_id,
    assemble_sequence,
    decode_coefficient,
    decode_sequence,
    encode_coefficient,
    encode_word,
    token_type_of,
)

# ---------------------------------------------------------------------------
# Vocabulary layout
# ---------------------------------------------------------------------------
class TestVocabulary:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 1012

    def test_word_band(self):
        assert token_type_of(WORD_OFFSET + 0) == 0  # a
        assert token_type_of(WORD_OFFSET + 5) == 0  # f

    def test_control_band(self):
        assert token_type_of(CONTROL_OFFSET + 0) == 1  # BOS
        assert token_type_of(CONTROL_OFFSET + CONTROL_SIZE - 1) == 1  # PAD
        assert token_type_of(COEFF_OFFSET) == 2        # PLUS_SIGN — now in coeff band

    def test_coeff_band(self):
        assert token_type_of(COEFF_OFFSET + 0) == 2    # PLUS_SIGN
        assert token_type_of(COEFF_OFFSET + 1) == 2    # MINUS_SIGN
        assert token_type_of(COEFF_OFFSET + COEFF_SIZE - 1) == 2  # NUM_999

    def test_bos_id(self):
        assert BOS_ID == 6

    def test_pad_id(self):
        assert PAD_ID == 9

    def test_sign_ids(self):
        assert PLUS_ID == 10
        assert MINUS_ID == 11
        assert token_type_of(PLUS_ID) == 2  # sign in coeff band
        assert token_type_of(MINUS_ID) == 2


# ---------------------------------------------------------------------------
# Coefficient encode/decode
# ---------------------------------------------------------------------------
class TestCoefficientEncode:
    def test_positive_small(self):
        assert encode_coefficient(5) == [PLUS_ID, num_id(5)]

    def test_negative_small(self):
        assert encode_coefficient(-7) == [MINUS_ID, num_id(7)]

    def test_zero(self):
        assert encode_coefficient(0) == [PLUS_ID, num_id(0)]

    def test_positive_multi_block(self):
        # 12334 = 12*1000 + 334
        assert encode_coefficient(12334) == [
            PLUS_ID, num_id(12), num_id(334)
        ]

    def test_negative_multi_block(self):
        assert encode_coefficient(-12334) == [
            MINUS_ID, num_id(12), num_id(334)
        ]

    def test_max_magnitude_from_audit(self):
        """860,160 = 860*1000 + 160 → [sign, NUM_860, NUM_160]"""
        tokens = encode_coefficient(860160)
        assert len(tokens) == 3  # sign + 2 blocks

    def test_min_value_from_audit(self):
        tokens = encode_coefficient(-860160)
        assert tokens[0] == MINUS_ID
        assert len(tokens) == 3


class TestCoefficientDecode:
    def test_positive(self):
        assert decode_coefficient([PLUS_ID, num_id(42)]) == 42

    def test_negative(self):
        assert decode_coefficient([MINUS_ID, num_id(99)]) == -99

    def test_zero(self):
        assert decode_coefficient([PLUS_ID, num_id(0)]) == 0

    def test_multi_block(self):
        assert decode_coefficient(
            [PLUS_ID, num_id(12), num_id(334)]
        ) == 12334

    def test_none_on_invalid_sign(self):
        # num_id(42) is not a sign token
        assert decode_coefficient([num_id(42), num_id(5)]) is None

    def test_none_on_empty(self):
        assert decode_coefficient([]) is None

    def test_none_on_too_short(self):
        assert decode_coefficient([PLUS_ID]) is None

    def test_stops_at_eos(self):
        assert decode_coefficient([PLUS_ID, num_id(5), EOS_ID, num_id(99)]) == 5

    def test_stops_at_pad(self):
        assert decode_coefficient([MINUS_ID, num_id(3), PAD_ID]) == -3


class TestCoefficientRoundTrip:
    @pytest.mark.parametrize("coeff", [
        0, 1, -1, 42, -99,
        999, 1000, 12334, -12334,
        100000, -100000,
        860160, -860160,  # max |c| from audit
    ])
    def test_round_trip(self, coeff):
        tokens = encode_coefficient(coeff)
        assert decode_coefficient(tokens) == coeff, \
            f"Round-trip failed for {coeff}: tokens={tokens}"


# ---------------------------------------------------------------------------
# Word encode
# ---------------------------------------------------------------------------
class TestWordEncode:
    def test_encode(self):
        assert encode_word("abcdef") == [0, 1, 2, 3, 4, 5]

    def test_encode_reversed(self):
        assert encode_word("fedcba") == [5, 4, 3, 2, 1, 0]


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------
class TestSequenceAssembly:
    def test_length(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        assert len(seq.tokens) == 32
        assert len(seq.token_types) == 32
        assert len(seq.loss_weights) == 32

    def test_bos_at_start(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        assert seq.tokens[0].item() == BOS_ID
        assert seq.loss_weights[0].item() == 0.0

    def test_word_positions(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        # positions 1-10 are word letters, all type 0, unsupervised
        for i in range(1, 11):
            assert seq.token_types[i].item() == 0, f"pos {i} not word type"
            assert seq.loss_weights[i].item() == 0.0, f"pos {i} supervised"

    def test_coeff_separator(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        assert seq.tokens[11].item() == COEFF_ID
        assert seq.loss_weights[11].item() == 0.0  # COEFF not supervised

    def test_sign_supervised(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        sign_pos = 12
        assert seq.tokens[sign_pos].item() in (PLUS_ID, MINUS_ID)
        assert seq.loss_weights[sign_pos].item() == 1.0

    def test_coeff_blocks_supervised(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        # After sign, find EOS
        for i in range(13, 32):
            if seq.tokens[i].item() == EOS_ID:
                break
            assert seq.loss_weights[i].item() == 1.0

    def test_eos_supervised(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        eos_positions = (seq.tokens == EOS_ID).nonzero(as_tuple=True)[0]
        assert len(eos_positions) == 1
        assert seq.loss_weights[eos_positions[0]].item() == 1.0

    def test_pad_unsupervised(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        pad_positions = (seq.tokens == PAD_ID).nonzero(as_tuple=True)[0]
        assert len(pad_positions) > 0
        for p in pad_positions:
            assert seq.loss_weights[p].item() == 0.0

    def test_pad_all_same_after_eos(self):
        seq = assemble_sequence("ababababab", 42, sequence_len=32)
        eos_pos = (seq.tokens == EOS_ID).nonzero(as_tuple=True)[0][0].item()
        for i in range(eos_pos + 1, 32):
            assert seq.tokens[i].item() == PAD_ID

    def test_raises_on_too_short(self):
        with pytest.raises(ValueError, match="too small"):
            assemble_sequence("ababababab", 42, sequence_len=10)

    def test_multi_block_coefficient(self):
        """12334 needs 2 blocks: sign + 2 blocks + EOS."""
        seq = assemble_sequence("ababababab", 12334, sequence_len=32)
        # Find EOS — should be at position 15 (1 BOS + 10 word + 1 COEFF + 1 sign + 2 blocks)
        eos_pos = (seq.tokens == EOS_ID).nonzero(as_tuple=True)[0][0].item()
        assert eos_pos == 15, f"EOS at {eos_pos}, expected 15"

    def test_negative_coefficient_loss_mask(self):
        seq = assemble_sequence("ababababab", -500, sequence_len=32)
        sign_pos = 12
        assert seq.tokens[sign_pos].item() == MINUS_ID
        assert seq.loss_weights[sign_pos].item() == 1.0


# ---------------------------------------------------------------------------
# Sequence decode
# ---------------------------------------------------------------------------
class TestSequenceDecode:
    def test_round_trip(self):
        word, coeff = "ababccdeff", 12334
        seq = assemble_sequence(word, coeff, sequence_len=32)
        result = decode_sequence(seq.tokens)
        assert result["valid"]
        assert result["word"] == word
        assert result["coefficient"] == coeff

    def test_round_trip_negative(self):
        word, coeff = "aabbccddee", -98765
        seq = assemble_sequence(word, coeff, sequence_len=32)
        result = decode_sequence(seq.tokens)
        assert result["valid"]
        assert result["coefficient"] == coeff

    def test_round_trip_zero(self):
        word, coeff = "ffffffffff", 0
        seq = assemble_sequence(word, coeff, sequence_len=32)
        result = decode_sequence(seq.tokens)
        assert result["valid"]
        assert result["coefficient"] == 0


# ---------------------------------------------------------------------------
# Loss mask contract
# ---------------------------------------------------------------------------
class TestLossMaskContract:
    """Verify that the loss mask matches the specification:
    - input word: NOT supervised
    - sign, base-1000 blocks, EOS: supervised
    - PAD: NOT supervised
    """

    def test_only_specified_positions_supervised(self):
        seq = assemble_sequence("abcdefabab", 12345, sequence_len=32)
        eos_pos = (seq.tokens == EOS_ID).nonzero(as_tuple=True)[0][0].item()

        for i in range(32):
            if i == 0:
                assert seq.loss_weights[i].item() == 0.0  # BOS
            elif 1 <= i <= 10:
                assert seq.loss_weights[i].item() == 0.0  # word
            elif i == 11:
                assert seq.loss_weights[i].item() == 0.0  # COEFF
            elif 12 <= i <= eos_pos:
                assert seq.loss_weights[i].item() == 1.0, f"pos {i} should be supervised"
            else:
                assert seq.loss_weights[i].item() == 0.0  # PAD
