"""Versioned heptagon letter IDs, independent of file encounter order."""

ALPHABET_VERSION = "heptagon-aij-row-major-v1"
LETTERS = tuple(f"a{i}{j}" for i in range(1, 7) for j in range(1, 8))
LETTER_TO_ID = {letter: index for index, letter in enumerate(LETTERS)}
WXF_LETTER_TO_ID = {f"Global`{letter}": index for letter, index in LETTER_TO_ID.items()}
WORD_LENGTH = 6
