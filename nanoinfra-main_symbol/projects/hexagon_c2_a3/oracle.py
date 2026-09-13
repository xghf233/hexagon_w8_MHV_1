"""Exact 0y oracle becomes available only after complete source validation."""

from pathlib import Path
import json

from . import audit_representation as audit
from .alphabet import ALPHABET_VERSION, TRAINING_ALIASES
from .data_contract import PHYSICAL_OBJECT, require


class ExactOracle:
    def __init__(self, source):
        self.source = Path(source).expanduser().resolve(strict=True)
        self._zero_y, self.targets, self.provenance = audit.read_source(self.source)
        manifest = json.loads((self.source.parent / "scaling_manifest.json").read_text(encoding="utf-8"))
        require(manifest["physical_object"] == PHYSICAL_OBJECT
                and manifest["physical_normalization_changed"] is False
                and manifest["alphabet_conversion_applied"] is False
                and manifest["alphabet_version"] == ALPHABET_VERSION
                and manifest["training_aliases"] == list(TRAINING_ALIASES),
                "Physical object, normalization or alphabet contract mismatch")

    def coefficient(self, word):
        require(len(word) == 8 and all(0 <= x < 6 for x in word), "Oracle accepts only legal 0y words")
        # Missing support really means zero only because the complete source passed.
        return self._zero_y.get(bytes(word), 0)
