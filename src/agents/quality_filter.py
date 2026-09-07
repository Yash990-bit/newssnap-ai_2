"""NewsSnap AI - Content Quality Filter Agent (Issue 12).

Filters scraped articles before they enter the summarization pipeline:
- Quality scoring on: content length, title/body coherence, completeness
- Clickbait detector: sensational title patterns vs. thin body content
- Sponsored content filter: keyword + pattern matching
- Duplicate press release detector: boilerplate PR text patterns
- Configurable threshold (default 0.4 out of 1.0)
- Logs rejected articles with article_id, source, rejection_reason, score
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clickbait patterns (title-level)
# ---------------------------------------------------------------------------
CLICKBAIT_TITLE_PATTERNS = [
    r"\byou won't believe\b",
    r"\bshocking\b",
    r"\bwhat happened next\b",
    r"\bblew.*mind\b",
    r"\bmust[- ]?see\b",
    r"\bviral\b",
    r"\bOMG\b",
    r"\bunbelievable\b",
    r"\bsecret revealed\b",
    r"!!!+",
    r"\b\d+\s+reasons\b",
    r"\bthis is why\b",
    r"\bwow\b",
    r"\bbreaking!!!+\b",
]

# ---------------------------------------------------------------------------
# Sponsored / advertorial markers
# ---------------------------------------------------------------------------
SPONSORED_PATTERNS = [
    r"\bsponsored\b",
    r"\bpartner content\b",
    r"\badvertorial\b",
    r"\bbrought to you by\b",
    r"\bpaid post\b",
    r"\bpromoted content\b",
    r"\bpaid partnership\b",
    r"\baffiliate\b",
]

# ---------------------------------------------------------------------------
# Press-release boilerplate markers
# ---------------------------------------------------------------------------
PRESS_RELEASE_PATTERNS = [
    r"\bfor immediate release\b",
    r"\bpress release\b",
    r"\bfor more information contact\b",
    r"\bmedia contact\b",
    r"\bpr newswire\b",
    r"\bbusiness wire\b",
    r"\bglobe newswire\b",
    r"\bends\b[\s\n]*$",  # typical PR sign-off
]

MIN_BODY_WORDS = 100  # AC: body < 100 words → rejected as incomplete


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RejectionRecord:
    """A logged rejection entry."""

    article_id: Optional[str]
    source: Optional[str]
    rejection_reason: str
    score: float
    title: Optional[str] = None


@dataclass
class FilterResult:
    """Result of filtering a single article."""

    passed: bool
    score: float
    rejection_reason: Optional[str] = None
    breakdown: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main QualityFilter class
# ---------------------------------------------------------------------------


class QualityFilter:
    """Score and filter news articles before summarization.

    Scoring dimensions (each 0.0–1.0, weighted average):
        - content_length   : word count relative to target (200+ words → 1.0)
        - clickbait        : 1.0 if no clickbait in title, else 0.0
        - coherence        : title keywords present in body (0.0–1.0)
        - completeness     : passes minimum word count + no ad/caption-only body
        - sponsored        : 1.0 if no sponsored markers found, else 0.0
    """

    WEIGHTS: Dict[str, float] = {
        "content_length": 0.30,
        "clickbait": 0.25,
        "coherence": 0.20,
        "completeness": 0.15,
        "sponsored": 0.10,
    }

    def __init__(self, threshold: float = 0.4):
        """
        Args:
            threshold: Minimum quality score [0.0, 1.0] to pass. Default 0.4.
        """
        self.threshold = threshold
        self._rejection_log: List[RejectionRecord] = []

    # ------------------------------------------------------------------
    # Individual dimension scorers
    # ------------------------------------------------------------------

    def _score_content_length(self, body: str) -> float:
        """Score based on word count. 200+ words → 1.0, linear below."""
        word_count = len(body.split())
        if word_count >= 200:
            return 1.0
        return word_count / 200.0

    def _score_clickbait(self, title: str) -> float:
        """1.0 if no clickbait detected in title, 0.0 if clickbait found."""
        lower = title.lower()
        for pattern in CLICKBAIT_TITLE_PATTERNS:
            if re.search(pattern, lower):
                return 0.0
        return 1.0

    def _score_coherence(self, title: str, body: str) -> float:
        """Fraction of meaningful title keywords found in body."""
        # Strip stopwords via a simple length filter (words > 3 chars)
        stop = {
            "the",
            "and",
            "for",
            "are",
            "was",
            "were",
            "has",
            "had",
            "have",
            "this",
            "that",
            "with",
            "from",
            "into",
            "after",
            "been",
            "also",
            "its",
            "will",
            "can",
            "but",
            "not",
        }
        title_words = [w.lower().strip(".,!?\"'") for w in title.split() if len(w) > 3 and w.lower() not in stop]
        if not title_words:
            return 1.0  # can't penalise if title has no meaningful words
        body_lower = body.lower()
        matches = sum(1 for w in title_words if w in body_lower)
        return min(matches / len(title_words), 1.0)

    def _score_completeness(self, body: str) -> float:
        """1.0 if body has ≥ MIN_BODY_WORDS and meaningful content, else 0.0."""
        word_count = len(body.split())
        if word_count < MIN_BODY_WORDS:
            return 0.0
        # Check for caption/ad-only bodies: very short unique sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", body) if len(s.strip()) > 10]
        if len(sentences) < 3:
            return 0.0
        return 1.0

    def _score_sponsored(self, body: str, title: str) -> float:
        """1.0 if no sponsored/advertorial markers, 0.0 if detected."""
        combined = (title + " " + body).lower()
        for pattern in SPONSORED_PATTERNS:
            if re.search(pattern, combined):
                return 0.0
        return 1.0

    def _is_press_release(self, body: str) -> bool:
        """True if body contains press release boilerplate."""
        lower = body.lower()
        for pattern in PRESS_RELEASE_PATTERNS:
            if re.search(pattern, lower):
                return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, article: Dict[str, Any]) -> float:
        """Compute a quality score [0.0, 1.0] for an article dict.

        Args:
            article: Dict with at least 'body' key; optionally 'title'.

        Returns:
            Weighted quality score between 0.0 and 1.0.
        """
        title = article.get("title", "") or ""
        body = article.get("body", "") or ""

        breakdown = self._compute_breakdown(title, body)
        total = sum(score * self.WEIGHTS[dim] for dim, score in breakdown.items())
        return round(total, 4)

    def _compute_breakdown(self, title: str, body: str) -> Dict[str, float]:
        return {
            "content_length": self._score_content_length(body),
            "clickbait": self._score_clickbait(title),
            "coherence": self._score_coherence(title, body),
            "completeness": self._score_completeness(body),
            "sponsored": self._score_sponsored(body, title),
        }

    def filter(
        self,
        article: Dict[str, Any],
    ) -> FilterResult:
        """Filter a single article.

        Returns a FilterResult with passed=True/False, score, and reason.
        Logs rejections internally.
        """
        title = article.get("title", "") or ""
        body = article.get("body", "") or ""
        article_id = article.get("article_id") or article.get("id")
        source = article.get("source")

        breakdown = self._compute_breakdown(title, body)
        total_score = round(sum(score * self.WEIGHTS[dim] for dim, score in breakdown.items()), 4)

        # Determine rejection reason (fast-fail checks first)
        rejection_reason = self._rejection_reason(title, body, total_score, breakdown)

        if rejection_reason:
            record = RejectionRecord(
                article_id=str(article_id) if article_id else None,
                source=source,
                rejection_reason=rejection_reason,
                score=total_score,
                title=title[:120] if title else None,
            )
            self._rejection_log.append(record)
            logger.info(
                "QualityFilter REJECTED article_id=%s source=%s score=%.3f reason=%s",
                article_id,
                source,
                total_score,
                rejection_reason,
            )
            return FilterResult(
                passed=False,
                score=total_score,
                rejection_reason=rejection_reason,
                breakdown=breakdown,
            )

        return FilterResult(passed=True, score=total_score, breakdown=breakdown)

    def _rejection_reason(
        self,
        title: str,
        body: str,
        score: float,
        breakdown: Dict[str, float],
    ) -> Optional[str]:
        """Return a human-readable rejection reason, or None if article passes."""
        word_count = len(body.split())

        # Hard rejection: sponsored content
        if breakdown["sponsored"] == 0.0:
            return "Sponsored/advertorial content detected"

        # Hard rejection: press release
        if self._is_press_release(body):
            return "Press release / boilerplate PR content detected"

        # Hard rejection: clickbait title
        if breakdown["clickbait"] == 0.0:
            return "Clickbait title detected"

        # Hard rejection: body too short
        if word_count < MIN_BODY_WORDS:
            return f"Incomplete: body has only {word_count} words (min {MIN_BODY_WORDS})"

        # Score below threshold
        if score < self.threshold:
            worst = min(breakdown, key=breakdown.get)  # type: ignore[arg-type]
            return (
                f"Quality score {score:.3f} below threshold {self.threshold:.3f} "
                f"(lowest dimension: {worst}={breakdown[worst]:.2f})"
            )

        return None

    def batch_filter(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[RejectionRecord]]:
        """Filter a batch of articles.

        Returns:
            (passed_articles, rejection_records)
        """
        before = len(self._rejection_log)
        passed = []
        for art in articles:
            result = self.filter(art)
            if result.passed:
                passed.append(art)
        new_rejections = self._rejection_log[before:]
        return passed, list(new_rejections)

    def get_rejection_log(self) -> List[RejectionRecord]:
        """Return all rejection records logged so far."""
        return list(self._rejection_log)

    def clear_log(self) -> None:
        """Clear the in-memory rejection log."""
        self._rejection_log.clear()
