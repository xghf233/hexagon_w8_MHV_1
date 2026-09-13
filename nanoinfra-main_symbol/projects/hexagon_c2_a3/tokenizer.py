"""C36 + ordered y types -> fixed-base100 target, with no physical position input."""

from dataclasses import dataclass
from numbers import Integral
import torch

from core.data.supervision import NextTokenPrediction
from core.tokenization.vocab_layout import VocabLayout
from .alphabet import ALPHABET_VERSION

TOKENIZER_VERSION = "hexagon-m2-adjacent38-base100-fixed2-packed128-v1"
BOS_ID, QUERY_ID, CONTEXT_ID, ANSWER_ID, EOS_ID, PAD_ID = range(9, 15)
PLUS_ID, MINUS_ID, NUM_OFFSET = 15, 16, 17
BASE, FIXED_BLOCKS = 100, 2
VOCAB_SIZE, N_TOKEN_TYPES = 117, 3
CONDITION_START, CONDITION_END = 5, 113
ANSWER_POSITION, PROMPT_LENGTH, ANSWER_LENGTH, SEQUENCE_LENGTH = 113, 114, 4, 128
IGNORE_INDEX = VocabLayout.IGNORE_INDEX


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _ids(values):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("Expected a one-dimensional integer sequence")
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        raise ValueError("Expected integer sequence, not text/bytes")
    try:
        return [_integer(value, "value") for value in values]
    except TypeError as exc:
        raise ValueError("Expected a one-dimensional integer sequence") from exc


def build_layout():
    layout = VocabLayout()
    for kind, start, end in ((0, 0, 9), (1, 9, 15), (2, 15, 117)):
        layout.add_range(kind, start, end)
    return layout


def token_type_of(token_id):
    token_id = _integer(token_id, "token ID")
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValueError("Token ID outside M2 vocabulary")
    return 0 if token_id < 9 else 1 if token_id < 15 else 2


def encoding_metadata(sequence_len=SEQUENCE_LENGTH):
    if _integer(sequence_len, "sequence_len") != SEQUENCE_LENGTH:
        raise ValueError("M2/38 requires sequence_len=128")
    return {
        "tokenizer_version": TOKENIZER_VERSION, "alphabet_version": ALPHABET_VERSION,
        "vocab_size": VOCAB_SIZE, "n_token_types": N_TOKEN_TYPES,
        "integer_base": BASE, "fixed_blocks": FIXED_BLOCKS, "coefficient_scale": 32,
        "logical_inputs": 38, "y_count": 2, "adjacent_only": True,
        "positions_in_prompt": False, "word_in_prompt": False,
        "sequence_len": SEQUENCE_LENGTH, "prompt_length": PROMPT_LENGTH,
        "answer_length": ANSWER_LENGTH, "condition_range": [5, 113],
        "supervised_shifted_range": [113, 117],
        "BOS": BOS_ID, "QUERY": QUERY_ID, "CONTEXT": CONTEXT_ID, "ANSWER": ANSWER_ID,
        "EOS": EOS_ID, "PAD": PAD_ID, "PLUS": PLUS_ID, "MINUS": MINUS_ID, "NUM_OFFSET": NUM_OFFSET,
    }


def encode_coefficient(coefficient):
    coefficient = _integer(coefficient, "coefficient")
    if abs(coefficient) >= BASE ** FIXED_BLOCKS:
        raise ValueError("Fixed-two base100 requires |C4| < 10000")
    high, low = divmod(abs(coefficient), BASE)
    return [MINUS_ID if coefficient < 0 else PLUS_ID, NUM_OFFSET + high, NUM_OFFSET + low]


def decode_coefficient(tokens):
    ids = _ids(tokens)
    if len(ids) != 3 or ids[0] not in (PLUS_ID, MINUS_ID):
        raise ValueError("Expected sign and exactly two magnitude blocks")
    if any(not NUM_OFFSET <= i < VOCAB_SIZE for i in ids[1:]):
        raise ValueError("Invalid numeric token")
    value = BASE * (ids[1] - NUM_OFFSET) + ids[2] - NUM_OFFSET
    if not value and ids[0] == MINUS_ID:
        raise ValueError("Negative zero is not canonical")
    return -value if ids[0] == MINUS_ID else value


def decode_generated(tokens):
    ids = _ids(tokens)
    if len(ids) < ANSWER_LENGTH or ids[3] != EOS_ID or any(i != PAD_ID for i in ids[4:]):
        raise ValueError("Expected sign/high/low/EOS, followed only by PAD")
    return decode_coefficient(ids[:3])


def encode_prompt(conditions, y_types):
    """The narrow inference API deliberately has no position, word or label argument."""
    conditions, y_types = _ids(conditions), _ids(y_types)
    if len(conditions) != 36:
        raise ValueError("Keep all 36 conditions, including zero and repeated coefficients")
    if len(y_types) != 2 or any(y not in (6, 7, 8) for y in y_types):
        raise ValueError("Expected two ordered native y IDs")
    numbers = [token for c in conditions for token in encode_coefficient(c)]
    return [BOS_ID, QUERY_ID, *y_types, CONTEXT_ID, *numbers, ANSWER_ID]


@dataclass
class AssembledSequence:
    tokens: torch.Tensor
    token_types: torch.Tensor
    loss_weights: torch.Tensor


def assemble_sequence(conditions, y_types, coefficient, sequence_len=SEQUENCE_LENGTH):
    encoding_metadata(sequence_len)
    tokens = encode_prompt(conditions, y_types) + encode_coefficient(coefficient) + [EOS_ID]
    tokens += [PAD_ID] * (sequence_len - len(tokens))
    weights = [0.0] * sequence_len
    weights[PROMPT_LENGTH:PROMPT_LENGTH + ANSWER_LENGTH] = [1.0] * ANSWER_LENGTH
    return AssembledSequence(
        torch.tensor(tokens, dtype=torch.long, device="cpu"),
        torch.tensor([token_type_of(t) for t in tokens], dtype=torch.long, device="cpu"),
        torch.tensor(weights, dtype=torch.float32, device="cpu"))


def assemble_batch(conditions, y_types, coefficients, sequence_len=SEQUENCE_LENGTH):
    """Necessary host preprocessing; copies mmap arrays and never evaluates a model."""
    encoding_metadata(sequence_len)
    conditions = torch.tensor(conditions, device="cpu")
    y_types = torch.tensor(y_types, device="cpu")
    coefficients = torch.tensor(coefficients, device="cpu")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if any(t.dtype not in integer_dtypes for t in (conditions, y_types, coefficients)):
        raise ValueError("Model inputs and labels must have integer dtype")
    if conditions.ndim != 2 or conditions.shape[1] != 36:
        raise ValueError("Conditions must have shape [N,36]")
    n = conditions.shape[0]
    if y_types.shape != (n, 2) or coefficients.shape != (n,):
        raise ValueError("Expected y_types [N,2] and labels [N]")
    conditions, y_types, coefficients = conditions.long(), y_types.long(), coefficients.long()
    if torch.any((y_types < 6) | (y_types > 8)):
        raise ValueError("Illegal y query ID")
    for values in (conditions, coefficients):
        if torch.any((values <= -10000) | (values >= 10000)):
            raise ValueError("Fixed-two base100 overflow")
    tokens = torch.full((n, sequence_len), PAD_ID, dtype=torch.long, device="cpu")
    tokens[:, 0], tokens[:, 1], tokens[:, 4] = BOS_ID, QUERY_ID, CONTEXT_ID
    tokens[:, 2:4] = y_types
    triples = torch.stack((torch.where(conditions < 0, MINUS_ID, PLUS_ID),
                          NUM_OFFSET + conditions.abs() // BASE,
                          NUM_OFFSET + conditions.abs() % BASE), dim=-1)
    tokens[:, CONDITION_START:CONDITION_END] = triples.reshape(n, 108)
    tokens[:, ANSWER_POSITION] = ANSWER_ID
    tokens[:, PROMPT_LENGTH] = torch.where(coefficients < 0, MINUS_ID, PLUS_ID)
    tokens[:, PROMPT_LENGTH + 1] = NUM_OFFSET + coefficients.abs() // BASE
    tokens[:, PROMPT_LENGTH + 2] = NUM_OFFSET + coefficients.abs() % BASE
    tokens[:, PROMPT_LENGTH + 3] = EOS_ID
    weights = torch.zeros_like(tokens, dtype=torch.float32)
    weights[:, PROMPT_LENGTH:PROMPT_LENGTH + ANSWER_LENGTH] = 1
    return AssembledSequence(tokens, build_layout().classify_token_types(tokens), weights)


def next_token_batch(sequence):
    if sequence.tokens.ndim != 2 or sequence.tokens.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("Expected batched 128-token sequences")
    return NextTokenPrediction().apply(sequence.tokens, sequence.token_types,
                                      loss_weights=sequence.loss_weights)
