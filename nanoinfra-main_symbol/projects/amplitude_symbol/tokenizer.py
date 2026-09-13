"""Fixed vocabulary and base-1000 encoding for five-loop symbol coefficients.

Vocabulary layout (3 bands, 1012 tokens total):

    Band 0 — word letters (type_id=0, offset=0, size=6):
        a=0, b=1, c=2, d=3, e=4, f=5

    Band 1 — control tokens (type_id=1, offset=6, size=4):
        BOS=6, COEFF=7, EOS=8, PAD=9

    Band 2 — coefficient band (type_id=2, offset=10, size=1002):
        PLUS_SIGN=10, MINUS_SIGN=11, NUM_0=12, ..., NUM_999=1011

Sign tokens live in the coefficient band so that all autoregressively
generated coefficient tokens share type_id=2, eliminating the train/
inference token-type mismatch.

Base-1000 encoding (matches the AIAmplitudes/paper convention):
    12334  -> [PLUS_SIGN, NUM_12, NUM_334]
    -12334 -> [MINUS_SIGN, NUM_12, NUM_334]
    0      -> [PLUS_SIGN, NUM_0]

The encode/decode functions operate on GLOBAL token IDs (not per-band local IDs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# Band layout constants
# ---------------------------------------------------------------------------
WORD_OFFSET = 0
WORD_SIZE = 6
CONTROL_OFFSET = 6
CONTROL_SIZE = 4
COEFF_OFFSET = 10
COEFF_SIZE = 1002

VOCAB_SIZE = 1012
N_TOKEN_TYPES = 3

# ---------------------------------------------------------------------------
# Per-band local token maps
# ---------------------------------------------------------------------------
WORD_TOKEN_IDS: dict[str, int] = {ch: i for i, ch in enumerate("abcdef")}
WORD_ID_TOKENS: dict[int, str] = {i: ch for ch, i in WORD_TOKEN_IDS.items()}

CONTROL_TOKEN_IDS: dict[str, int] = {
    "BOS": 0, "COEFF": 1, "EOS": 2, "PAD": 3,
}
CONTROL_ID_TOKENS: dict[int, str] = {i: name for name, i in CONTROL_TOKEN_IDS.items()}

# Coeff-band sign tokens (local indices within the coeff band)
COEFF_PLUS = 0   # local index for PLUS_SIGN in coeff band
COEFF_MINUS = 1  # local index for MINUS_SIGN in coeff band
COEFF_NUM_START = 2  # NUM_0 starts at local index 2

# ---------------------------------------------------------------------------
# Convenience globals (global IDs)
# ---------------------------------------------------------------------------
BOS_ID = CONTROL_OFFSET + CONTROL_TOKEN_IDS["BOS"]
COEFF_ID = CONTROL_OFFSET + CONTROL_TOKEN_IDS["COEFF"]
EOS_ID = CONTROL_OFFSET + CONTROL_TOKEN_IDS["EOS"]
PAD_ID = CONTROL_OFFSET + CONTROL_TOKEN_IDS["PAD"]
PLUS_ID = COEFF_OFFSET + COEFF_PLUS     # 10 — in coeff band
MINUS_ID = COEFF_OFFSET + COEFF_MINUS   # 11 — in coeff band


def num_id(value: int) -> int:
    """Return the global token ID for NUM_<value>."""
    if not (0 <= value < 1000):
        raise ValueError(f"NUM value {value} out of range [0, 999]")
    return COEFF_OFFSET + COEFF_NUM_START + value

IGNORE_INDEX = -1  # matches VocabLayout.IGNORE_INDEX


# ---------------------------------------------------------------------------
# Token type classifier (maps global ID -> type_id)
# ---------------------------------------------------------------------------
def token_type_of(global_id: int) -> int:
    """Return the token type (band index) for a global token ID."""
    if global_id < CONTROL_OFFSET:
        return 0  # word band
    elif global_id < COEFF_OFFSET:
        return 1  # control band
    else:
        return 2  # coefficient band (sign + numbers)


# ---------------------------------------------------------------------------
# Coefficient encode / decode
# ---------------------------------------------------------------------------
def encode_coefficient(coeff: int) -> list[int]:
    """Encode a signed integer into sign + base-1000 global token IDs.

    Sign tokens live in the coefficient band (type_id=2), so all
    generated tokens share the same type during autoregressive inference.

    >>> encode_coefficient(12334)
    [10, 24, 346]
    >>> encode_coefficient(-12334)
    [11, 24, 346]
    >>> encode_coefficient(0)
    [10, 12]
    """
    sign_id = PLUS_ID if coeff >= 0 else MINUS_ID
    abs_val = abs(coeff)

    if abs_val == 0:
        return [sign_id, COEFF_OFFSET + COEFF_NUM_START + 0]

    # Extract base-1000 blocks (least-significant first, then reversed)
    blocks: list[int] = []
    v = abs_val
    while v > 0:
        blocks.append(v % 1000)
        v //= 1000

    result = [sign_id]
    for b in reversed(blocks):
        result.append(COEFF_OFFSET + COEFF_NUM_START + b)
    return result


def decode_coefficient(token_ids: list[int]) -> int | None:
    """Decode a list of global token IDs back to a signed integer.

    Returns None if the token sequence is invalid or incomplete.
    Tokens after EOS or PAD are ignored.

    >>> decode_coefficient([10, 24, 346])
    12334
    >>> decode_coefficient([11, 24, 346])
    -12334
    >>> decode_coefficient([10, 12])
    0
    """
    if not token_ids:
        return None

    # Find where the coefficient definition ends
    end = len(token_ids)
    for i, tid in enumerate(token_ids):
        if tid in (EOS_ID, PAD_ID):
            end = i
            break

    coeff_tokens = token_ids[:end]
    if len(coeff_tokens) < 2:
        return None

    # First token must be sign
    sign_id = coeff_tokens[0]
    if sign_id not in (PLUS_ID, MINUS_ID):
        return None
    sign = 1 if sign_id == PLUS_ID else -1

    # Remaining tokens are base-1000 blocks
    abs_val = 0
    for tid in coeff_tokens[1:]:
        if not (COEFF_OFFSET + COEFF_NUM_START <= tid < COEFF_OFFSET + COEFF_SIZE):
            return None
        abs_val = abs_val * 1000 + (tid - COEFF_OFFSET - COEFF_NUM_START)

    return sign * abs_val


# ---------------------------------------------------------------------------
# Word encode / decode
# ---------------------------------------------------------------------------
def encode_word(word: str) -> list[int]:
    """Encode a word string into global token IDs.

    >>> encode_word("ababccdeff")
    [0, 1, 0, 1, 2, 2, 3, 4, 5, 5]
    """
    return [WORD_OFFSET + WORD_TOKEN_IDS[ch] for ch in word]


def decode_word(token_ids: list[int]) -> str | None:
    """Decode global token IDs back to a word string.

    Returns None if any ID is outside the word band.
    """
    chars = []
    for tid in token_ids:
        if not (WORD_OFFSET <= tid < WORD_OFFSET + WORD_SIZE):
            return None
        chars.append(WORD_ID_TOKENS[tid - WORD_OFFSET])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------
@dataclass
class AssembledSequence:
    """A fully assembled sequence ready for the DataSource."""
    tokens: torch.Tensor       # [L]  global token IDs
    token_types: torch.Tensor  # [L]  0=word, 1=control, 2=coeff
    loss_weights: torch.Tensor # [L]  0.0=unsupervised, 1.0=supervised


def assemble_sequence(
    word: str,
    coeff: int,
    sequence_len: int = 32,
) -> AssembledSequence:
    """Assemble a complete training sequence with loss mask.

    Sequence layout:
        [BOS][10×word][COEFF][sign][base-1000 blocks...][EOS][PAD...]

    Loss mask (supervised positions):
        - BOS, word letters, COEFF separator:  NOT supervised (loss_weight=0)
        - sign, base-1000 blocks, EOS:         supervised     (loss_weight=1)
        - PAD:                                  NOT supervised (loss_weight=0)

    Returns:
        AssembledSequence with all tensors of length `sequence_len`.
    """
    tokens: list[int] = []
    token_types: list[int] = []
    loss_weights: list[float] = []

    # 1. BOS
    tokens.append(BOS_ID)
    token_types.append(1)       # control
    loss_weights.append(0.0)

    # 2. Word letters (10 letters)
    for ch in word:
        tokens.append(WORD_OFFSET + WORD_TOKEN_IDS[ch])
        token_types.append(0)   # word
        loss_weights.append(0.0)

    # 3. COEFF separator
    tokens.append(COEFF_ID)
    token_types.append(1)
    loss_weights.append(0.0)

    # 4. Sign + base-1000 blocks (all supervised)
    coeff_tokens = encode_coefficient(coeff)
    for ct in coeff_tokens:
        tokens.append(ct)
        token_types.append(token_type_of(ct))
        loss_weights.append(1.0)

    # 5. EOS (supervised)
    tokens.append(EOS_ID)
    token_types.append(1)
    loss_weights.append(1.0)

    # 6. PAD to sequence_len
    n_pad = sequence_len - len(tokens)
    if n_pad < 0:
        raise ValueError(
            f"Sequence length {sequence_len} too small: need at least {len(tokens)} "
            f"(word={word}, coeff={coeff}, blocks={len(coeff_tokens)})"
        )
    for _ in range(n_pad):
        tokens.append(PAD_ID)
        token_types.append(1)
        loss_weights.append(0.0)

    return AssembledSequence(
        tokens=torch.tensor(tokens, dtype=torch.long),
        token_types=torch.tensor(token_types, dtype=torch.long),
        loss_weights=torch.tensor(loss_weights, dtype=torch.float32),
    )


def decode_sequence(tokens: torch.Tensor) -> dict:
    """Decode a full sequence tensor back to human-readable components.

    Returns dict with keys: word, coefficient, valid.
    `valid` is True if the sequence successfully decoded.
    """
    ids = tokens.tolist()

    # Find BOS
    try:
        bos_idx = ids.index(BOS_ID)
    except ValueError:
        return {"word": None, "coefficient": None, "valid": False}

    # Find COEFF after BOS
    rest = ids[bos_idx + 1:]
    try:
        coeff_idx_in_rest = rest.index(COEFF_ID)
    except ValueError:
        return {"word": None, "coefficient": None, "valid": False}

    # Word letters are between BOS and COEFF
    word_ids = rest[:coeff_idx_in_rest]
    word = decode_word(word_ids)

    # Find EOS after COEFF to delimit coefficient
    after_coeff = rest[coeff_idx_in_rest + 1:]
    try:
        eos_idx = after_coeff.index(EOS_ID)
    except ValueError:
        # No EOS — try to decode anyway up to PAD or end
        coeff_ids = []
        for tid in after_coeff:
            if tid == PAD_ID:
                break
            coeff_ids.append(tid)
    else:
        coeff_ids = after_coeff[:eos_idx]

    coefficient = decode_coefficient(coeff_ids)

    return {
        "word": word,
        "coefficient": coefficient,
        "valid": word is not None and coefficient is not None,
    }
