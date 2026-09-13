"""Eight native letters -> sign + TWO fixed base-100 blocks + EOS.

High NUM_0 is mandatory and distinct from PAD. No floating point coefficients.
"""

from dataclasses import dataclass
from numbers import Integral

import torch

from core.data.supervision import NextTokenPrediction
from core.tokenization.vocab_layout import VocabLayout
from .alphabet import ALPHABET_VERSION, LETTERS, LETTER_TO_ID, WORD_LENGTH, ZERO_Y_SIZE

TOKENIZER_VERSION = "hexagon-w8-base100-fixed2-v1"
WORD_START = 1
WORD_END = WORD_START + WORD_LENGTH  # exclusive; never include the answer
COEFF_POSITION = WORD_END
PROMPT_LENGTH = COEFF_POSITION + 1
SEQUENCE_LENGTH = 16
WORD_OFFSET, WORD_SIZE = 0, 9
CONTROL_OFFSET, CONTROL_SIZE = 9, 4
COEFF_OFFSET, COEFF_SIZE = 13, 102
BOS_ID, COEFF_ID, EOS_ID, PAD_ID = 9, 10, 11, 12
PLUS_ID, MINUS_ID, NUM_OFFSET = 13, 14, 15
BASE, FIXED_BLOCKS, ANSWER_LENGTH = 100, 2, 4
VOCAB_SIZE, N_TOKEN_TYPES = 115, 3
IGNORE_INDEX = VocabLayout.IGNORE_INDEX


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _ids(values) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("Expected one-dimensional token IDs")
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise ValueError("Expected a sequence of integer IDs, not a string")
    try:
        return [_integer(value, "token ID") for value in values]
    except TypeError as exc:
        raise ValueError("Expected a one-dimensional sequence of integer IDs") from exc


def validate_word_ids(word_ids) -> list[int]:
    ids = _ids(word_ids)
    if len(ids) != WORD_LENGTH or any(not 0 <= i < ZERO_Y_SIZE for i in ids):
        raise ValueError("Expected exactly eight 0y letter IDs in [0, 5]")
    return ids


def encode_word(letters) -> list[int]:
    """Encode eight native labels; concatenated text and y letters are rejected."""
    if isinstance(letters, (str, bytes)):
        raise ValueError("Pass eight separate native labels, not a concatenated string")
    try:
        ids = [LETTER_TO_ID[letter] for letter in letters]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unknown hexagon letter") from exc
    return validate_word_ids(ids)


def decode_word(word_ids) -> tuple[str, ...]:
    return tuple(LETTERS[i] for i in validate_word_ids(word_ids))


def num_id(value: int) -> int:
    value = _integer(value, "base-100 block")
    if not 0 <= value < BASE:
        raise ValueError("Base-100 block must be in [0, 99]")
    return NUM_OFFSET + value


def token_type_of(token_id: int) -> int:
    token_id = _integer(token_id, "token ID")
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValueError("Token ID outside hexagon vocabulary")
    return 0 if token_id < CONTROL_OFFSET else (1 if token_id < COEFF_OFFSET else 2)


def build_layout() -> VocabLayout:
    layout = VocabLayout()
    layout.add_range(0, WORD_OFFSET, CONTROL_OFFSET)
    layout.add_range(1, CONTROL_OFFSET, COEFF_OFFSET)
    layout.add_range(2, COEFF_OFFSET, VOCAB_SIZE)
    return layout


def encoding_metadata(sequence_len: int = SEQUENCE_LENGTH) -> dict:
    sequence_len = _integer(sequence_len, "sequence_len")
    if sequence_len != SEQUENCE_LENGTH:
        raise ValueError("This version requires sequence_len=16")
    return {
        "tokenizer_version": TOKENIZER_VERSION, "alphabet_version": ALPHABET_VERSION,
        "word_length": WORD_LENGTH, "vocab_size": VOCAB_SIZE,
        "n_token_types": N_TOKEN_TYPES, "integer_base": BASE,
        "fixed_blocks": FIXED_BLOCKS, "answer_length": ANSWER_LENGTH,
        "coefficient_scale": 32, "y_count": 0,
        "sequence_len": sequence_len, "prompt_length": PROMPT_LENGTH,
        "word_start": WORD_START, "word_end": WORD_END,
        "BOS": BOS_ID, "COEFF": COEFF_ID, "EOS": EOS_ID, "PAD": PAD_ID,
        "PLUS": PLUS_ID, "MINUS": MINUS_ID, "NUM_OFFSET": NUM_OFFSET,
    }


def encode_coefficient(coefficient: int) -> list[int]:
    coefficient = _integer(coefficient, "coefficient")
    sign = MINUS_ID if coefficient < 0 else PLUS_ID
    if abs(coefficient) >= BASE ** FIXED_BLOCKS:
        raise ValueError("Fixed-two base100 requires |C4| < 10000")
    high, low = divmod(abs(coefficient), BASE)
    return [sign, num_id(high), num_id(low)]


def decode_coefficient(tokens) -> int:
    """Decode sign + blocks only (no EOS/PAD); require canonical representation."""
    ids = _ids(tokens)
    if len(ids) != 3 or ids[0] not in (PLUS_ID, MINUS_ID):
        raise ValueError("Coefficient requires a sign and exactly two base100 blocks")
    if any(not NUM_OFFSET <= i < VOCAB_SIZE for i in ids[1:]):
        raise ValueError("Invalid coefficient magnitude token")
    blocks = [i - NUM_OFFSET for i in ids[1:]]
    value = BASE * blocks[0] + blocks[1]
    if value == 0 and ids[0] == MINUS_ID:
        raise ValueError("Negative zero is not canonical")
    return -value if ids[0] == MINUS_ID else value


def decode_generated(tokens) -> int:
    """Decode the generated suffix; EOS is mandatory and only PAD may follow it."""
    ids = _ids(tokens)
    if EOS_ID not in ids:
        raise ValueError("Missing EOS")
    end = ids.index(EOS_ID)
    if end != ANSWER_LENGTH - 1:
        raise ValueError("EOS must follow the sign and exactly two blocks")
    if any(i != PAD_ID for i in ids[end + 1:]):
        raise ValueError("Unexpected content after EOS")
    return decode_coefficient(ids[:end])


def encode_prompt(word_ids) -> list[int]:
    """Inference prefix only: no coefficient argument and no target tokens."""
    return [BOS_ID, *validate_word_ids(word_ids), COEFF_ID]


@dataclass
class AssembledSequence:
    tokens: torch.Tensor
    token_types: torch.Tensor
    loss_weights: torch.Tensor


def assemble_sequence(word_ids, coefficient: int,
                      sequence_len: int = SEQUENCE_LENGTH) -> AssembledSequence:
    encoding_metadata(sequence_len)
    answer = [*encode_coefficient(coefficient), EOS_ID]
    tokens = encode_prompt(word_ids) + answer
    if len(tokens) > sequence_len:
        raise ValueError(f"Sequence needs {len(tokens)} positions, got {sequence_len}")
    weights = [0.0] * PROMPT_LENGTH + [1.0] * len(answer)
    padding = sequence_len - len(tokens)
    tokens += [PAD_ID] * padding
    weights += [0.0] * padding
    return AssembledSequence(
        torch.tensor(tokens, dtype=torch.long, device="cpu"),
        torch.tensor([token_type_of(i) for i in tokens], dtype=torch.long, device="cpu"),
        torch.tensor(weights, dtype=torch.float32, device="cpu"),
    )


def decode_sequence(tokens) -> tuple[tuple[str, ...], int]:
    ids = _ids(tokens)
    if len(ids) != SEQUENCE_LENGTH or ids[0] != BOS_ID or ids[COEFF_POSITION] != COEFF_ID:
        raise ValueError("Invalid hexagon sequence prefix")
    return decode_word(ids[WORD_START:WORD_END]), decode_generated(ids[PROMPT_LENGTH:])


def assemble_batch(words, coefficients,
                   sequence_len: int = SEQUENCE_LENGTH) -> AssembledSequence:
    """Vectorized host preencoding for fixed-two-block integer coefficients.

    Copies array inputs: read-only mmap storage is never handed to torch for
    mutation. Zero is supported by the tokenizer, but forbidden by the dataset.
    """
    encoding_metadata(sequence_len)
    try:
        words = torch.tensor(words, device="cpu")
        coefficients = torch.tensor(coefficients, device="cpu")
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("Unsupported word/coefficient array representation") from exc
    for tensor in (words, coefficients):
        if tensor.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Words and coefficients must have integer dtype")
    if words.ndim != 2 or words.shape[1] != WORD_LENGTH:
        raise ValueError("Words must have shape [N, 8]")
    if coefficients.shape != (len(words),):
        raise ValueError("Coefficients must have shape [N]")
    words, coefficients = words.long(), coefficients.long()
    if torch.any((words < 0) | (words >= ZERO_Y_SIZE)):
        raise ValueError("0y letter ID outside [0, 5]")
    if torch.any((coefficients <= -(BASE ** FIXED_BLOCKS)) | (coefficients >= BASE ** FIXED_BLOCKS)):
        raise ValueError("Fixed-two base100 requires |C4| < 10000")
    tokens = torch.full((len(words), sequence_len), PAD_ID, dtype=torch.long, device="cpu")
    tokens[:, 0] = BOS_ID
    tokens[:, WORD_START:WORD_END] = words
    tokens[:, COEFF_POSITION] = COEFF_ID
    tokens[:, PROMPT_LENGTH] = torch.where(coefficients < 0, MINUS_ID, PLUS_ID)
    tokens[:, PROMPT_LENGTH + 1] = NUM_OFFSET + coefficients.abs() // BASE
    tokens[:, PROMPT_LENGTH + 2] = NUM_OFFSET + coefficients.abs() % BASE
    tokens[:, PROMPT_LENGTH + 3] = EOS_ID
    weights = torch.zeros_like(tokens, dtype=torch.float32)
    weights[:, PROMPT_LENGTH:PROMPT_LENGTH + ANSWER_LENGTH] = 1
    return AssembledSequence(tokens, build_layout().classify_token_types(tokens), weights)


def next_token_batch(sequence: AssembledSequence) -> dict[str, torch.Tensor]:
    """[B,L] -> [B,L-1]; supervise sign, magnitude blocks and EOS only."""
    if sequence.tokens.ndim != 2:
        raise ValueError("next_token_batch expects batched sequences")
    return NextTokenPrediction().apply(
        sequence.tokens, sequence.token_types, loss_weights=sequence.loss_weights,
    )
