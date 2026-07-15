"""Minimal ML fake news detector using a multinomial Naive Bayes model."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


Label = str
Sample = Tuple[str, Label]


@dataclass
class Prediction:
    label: Label
    confidence: float


class NaiveBayesFakeNewsDetector:
    """Simple text classifier for fake-vs-real news detection."""

    def __init__(self) -> None:
        self.labels: Sequence[Label] = ("fake", "real")
        self.class_doc_counts: Dict[Label, int] = Counter()
        self.class_word_counts: Dict[Label, Counter[str]] = defaultdict(Counter)
        self.class_total_words: Dict[Label, int] = Counter()
        self.vocabulary: set[str] = set()
        self._is_fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9']+", text.lower())

    def fit(self, samples: Iterable[Sample]) -> None:
        seen_any = False
        for text, label in samples:
            if label not in self.labels:
                raise ValueError(f"Unsupported label: {label}")

            seen_any = True
            tokens = self._tokenize(text)
            self.class_doc_counts[label] += 1
            self.class_word_counts[label].update(tokens)
            self.class_total_words[label] += len(tokens)
            self.vocabulary.update(tokens)

        if not seen_any:
            raise ValueError("At least one training sample is required")

        self._is_fitted = True

    def _log_prior(self, label: Label) -> float:
        total_docs = sum(self.class_doc_counts.values())
        return math.log(self.class_doc_counts[label] / total_docs)

    def _log_likelihood(self, tokens: List[str], label: Label) -> float:
        # Laplace smoothing
        vocab_size = max(1, len(self.vocabulary))
        total_words = self.class_total_words[label]
        word_counts = self.class_word_counts[label]

        score = 0.0
        for token in tokens:
            score += math.log((word_counts[token] + 1) / (total_words + vocab_size))
        return score

    def predict(self, text: str) -> Prediction:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        tokens = self._tokenize(text)
        log_scores = {
            label: self._log_prior(label) + self._log_likelihood(tokens, label)
            for label in self.labels
        }

        best_label = max(log_scores, key=log_scores.get)
        margin = abs(log_scores[self.labels[0]] - log_scores[self.labels[1]])
        confidence = max(0.5, min(0.99, 0.5 + margin / 10))
        return Prediction(label=best_label, confidence=confidence)


DEFAULT_TRAINING_DATA: Sequence[Sample] = (
    ("Breaking: celebrity says aliens control elections without evidence", "fake"),
    ("Miracle cure discovered overnight doctors hate this trick", "fake"),
    ("Shocking conspiracy claims government controls weather by secret machines", "fake"),
    ("Official health department releases weekly disease surveillance report", "real"),
    ("Parliament passes budget bill after public debate and voting", "real"),
    ("University publishes peer reviewed climate impact study", "real"),
    ("Viral post claims moon landing was filmed in basement", "fake"),
    ("Central bank announces interest rate decision in policy meeting", "real"),
)


def build_default_detector() -> NaiveBayesFakeNewsDetector:
    detector = NaiveBayesFakeNewsDetector()
    detector.fit(DEFAULT_TRAINING_DATA)
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal ML fake news detector")
    parser.add_argument("--text", required=True, help="News headline or short article")
    args = parser.parse_args()

    detector = build_default_detector()
    prediction = detector.predict(args.text)

    print(f"Prediction: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.2f}")


if __name__ == "__main__":
    main()
