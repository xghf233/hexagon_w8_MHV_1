"""Native hatted alphabet; no basis conversion and no second x32 scaling."""

ALPHABET_VERSION = "hexagon_hat_native_v1"
LETTERS = ("a", "b", "c", "mu", "mv", "mw", "yu", "yv", "yw")
TRAINING_ALIASES = ("hat_a", "hat_b", "hat_c", "hat_d", "hat_e", "hat_f", "y_U", "y_V", "y_W")
LETTER_TO_ID = {letter: i for i, letter in enumerate(LETTERS)}
WORD_LENGTH, ZERO_Y_SIZE, COEFFICIENT_SCALE = 8, 6, 32


def adjacent_positions(word):
    positions = tuple(i for i, letter in enumerate(word) if letter >= ZERO_Y_SIZE)
    return positions if len(positions) == 2 and positions[1] == positions[0] + 1 else None
