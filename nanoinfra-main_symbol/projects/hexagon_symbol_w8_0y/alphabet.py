"""One native symbol letter per token; no change of mathematical basis."""

ALPHABET_VERSION = "hexagon_hat_native_v1"
LETTERS = ("a", "b", "c", "mu", "mv", "mw", "yu", "yv", "yw")
TRAINING_ALIASES = ("hat_a", "hat_b", "hat_c", "hat_d", "hat_e", "hat_f", "y_U", "y_V", "y_W")
LETTER_TO_ID = {letter: i for i, letter in enumerate(LETTERS)}
WORD_LENGTH = 8
ZERO_Y_SIZE = 6
COEFFICIENT_SCALE = 32
