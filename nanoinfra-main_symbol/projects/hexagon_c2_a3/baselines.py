"""Train-only majority and exact input38 lookup; positions are NEVER part of the key."""

from collections import Counter
from .build_samples import input_key
from .data_contract import require
from .evaluator import checked_rows, metrics, prediction_report


class TrainBaselines:
    def __init__(self, train):
        require(train.split == "train", "Baseline fitting requires train")
        require(train.subset_protocol is None, "Fit baselines on the complete frozen train pool")
        self.metadata_sha256 = train.metadata_sha256
        self.table, histogram = {}, Counter()
        for c, y, label in zip(train.conditions[train.indices], train.y_types[train.indices],
                               train.coefficients[train.indices], strict=True):
            key, value = input_key(c, y), int(label)
            require(key not in self.table or self.table[key] == value, "Conflicting train lookup labels")
            self.table[key] = value
            histogram[value] += 1
        self.majority = min(histogram, key=lambda c: (-histogram[c], c))
        self.train_rows = len(train)

    def evaluate(self, dataset, rows):
        require(dataset.metadata_sha256 == self.metadata_sha256, "Baseline/evaluation dataset mismatch")
        rows = checked_rows(dataset, rows)
        keys = [input_key(c, y) for c, y in zip(dataset.conditions[rows], dataset.y_types[rows], strict=True)]
        covered = [key in self.table for key in keys]
        predictions = [self.table.get(key, self.majority) for key in keys]
        truths = dataset.coefficients[rows].tolist()
        covered_metrics = metrics([p for p, seen in zip(predictions, covered) if seen],
                                  [t for t, seen in zip(truths, covered) if seen])
        require(covered_metrics["n_samples"] == 0 or covered_metrics["exact_accuracy"] == 1,
                "A covered deterministic input has the wrong label")
        return {
            "fit_split": "complete train", "train_rows": self.train_rows,
            "train_unique_inputs": len(self.table), "majority_value": self.majority,
            "key": "(ordered_C36, ordered_y_types); no positions/word/labels",
            "lookup_coverage": {"n_samples": len(rows), "n_covered": sum(covered),
                                "fraction": sum(covered) / len(rows), "covered_metrics": covered_metrics},
            "majority": prediction_report(dataset, rows, [self.majority] * len(rows)),
            "lookup_with_majority_fallback": prediction_report(dataset, rows, predictions),
        }
