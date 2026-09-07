"""Tests for Content Quality Filter (Issue 12).

Acceptance Criteria covered:
  AC1 - Quality score computed on: content length, title/body coherence,
        source reliability, completeness
  AC2 - Clickbait articles rejected (sensational title, thin body)
  AC3 - Articles with body < 100 words rejected as incomplete
  AC4 - Sponsored content detected and filtered (keyword + pattern matching)
  AC5 - Quality threshold is configurable (default: 0.4 out of 1.0)
  AC6 - Rejected articles logged with: article_id, source, rejection_reason, score
  AC7 - GET /api/admin/rejected-articles returns recent rejections with reasons
  AC8 - Filter processes 100 articles in < 10 seconds
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient
from src.agents.quality_filter import (
    MIN_BODY_WORDS,
    QualityFilter,
    RejectionRecord,
)
from src.api.main import app

# ---------------------------------------------------------------------------
# Sample article bodies
# ---------------------------------------------------------------------------

GOOD_BODY = (
    "The Reserve Bank of India on Friday cut the repo rate by 25 basis points to 6.25 percent, "
    "the first rate reduction since February 2023. RBI Governor Shaktikanta Das announced the "
    "decision following the Monetary Policy Committee meeting in Mumbai. The move is expected "
    "to reduce home loan EMIs by approximately Rs 1,500 per month on a Rs 50 lakh loan. "
    "Stock markets surged following the announcement with the Sensex gaining 450 points to "
    "close at 79,800. Economists expect GDP growth to accelerate to 7.2 percent in the coming "
    "fiscal year as borrowing costs decline across sectors. The decision was unanimous among "
    "all six MPC members, marking a significant policy shift after 18 months of unchanged rates. "
    "Banking sector analysts noted that the transmission to retail lending rates may take two to "
    "three months. The central bank also revised its inflation forecast downward to 4.5 percent."
)  # ~130 words

GOOD_TITLE = "RBI cuts repo rate by 25 bps to 6.25% in first reduction since 2023"

SHORT_BODY = "India won the match."  # < 100 words — incomplete

CLICKBAIT_TITLE = "You WON'T BELIEVE what the RBI did next!!!"
CLICKBAIT_BODY = GOOD_BODY  # good body, bad title

SPONSORED_BODY = (
    "Sponsored content: This promotional article is brought to you by XYZ Bank. "
    "The Reserve Bank of India on Friday cut the repo rate by 25 basis points to 6.25 percent. "
    "RBI Governor Shaktikanta Das announced the decision following the MPC meeting. "
    "Stock markets surged following the announcement with the Sensex gaining 450 points to close higher. "
    "Economists expect GDP growth to accelerate to 7.2 percent in the coming fiscal year as borrowing costs decline. "
    "Home loan borrowers will benefit from lower EMIs starting next quarter, commercial banks confirmed today. "
    "Analysts widely welcome the shift in monetary policy stance from neutral to accommodative across the board. "
    "Banking sector experts also highlighted that inflation forecasts have moderated significantly in recent months."
)

PR_BODY = (
    "FOR IMMEDIATE RELEASE. XYZ Corporation today announced its record quarterly revenue for Q3 2024. "
    "The Reserve Bank of India on Friday cut the repo rate by 25 basis points to 6.25 percent. "
    "RBI Governor Shaktikanta Das announced the policy decision after the MPC meeting in Mumbai. "
    "Stock markets surged following the announcement with Sensex gaining 450 points, closing at 79,800. "
    "For more information contact: pr@xyzcorp.com. Media contact: Jane Doe, phone number 9999999999. "
    "Economists expect GDP growth to accelerate to 7.2 percent in the coming fiscal year. "
    "Lower borrowing costs are expected to benefit consumers and commercial businesses alike across India. "
    "The company will continue expanding its retail presence throughout the current fiscal year."
)


def make_good_article(**overrides) -> dict:
    base = {
        "article_id": "art-001",
        "source": "The Hindu",
        "title": GOOD_TITLE,
        "body": GOOD_BODY,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AC1 - Quality score computed on multiple dimensions
# ---------------------------------------------------------------------------


class TestQualityScoring:
    def test_score_returns_float_between_0_and_1(self):
        """AC1: score() returns a float in [0.0, 1.0]."""
        qf = QualityFilter()
        score = qf.score(make_good_article())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_good_article_scores_above_threshold(self):
        """AC1: A high-quality article scores above the default threshold."""
        qf = QualityFilter()
        score = qf.score(make_good_article())
        assert score >= qf.threshold, f"Good article scored {score}, expected >= {qf.threshold}"

    def test_short_article_scores_lower_than_long(self):
        """AC1: Shorter body → lower content_length dimension → lower total score."""
        qf = QualityFilter()
        short_score = qf.score(make_good_article(body="India won. " * 12))  # ~24 words
        long_score = qf.score(make_good_article())
        assert short_score < long_score

    def test_breakdown_contains_all_dimensions(self):
        """AC1: Internal breakdown has all scoring dimensions."""
        qf = QualityFilter()
        title = GOOD_TITLE
        body = GOOD_BODY
        breakdown = qf._compute_breakdown(title, body)
        expected_dims = {"content_length", "clickbait", "coherence", "completeness", "sponsored"}
        assert set(breakdown.keys()) == expected_dims

    def test_each_dimension_between_0_and_1(self):
        """AC1: Every individual dimension score is in [0.0, 1.0]."""
        qf = QualityFilter()
        breakdown = qf._compute_breakdown(GOOD_TITLE, GOOD_BODY)
        for dim, val in breakdown.items():
            assert 0.0 <= val <= 1.0, f"Dimension {dim} out of range: {val}"

    def test_coherent_title_body_scores_high_coherence(self):
        """AC1: Title keywords present in body → high coherence score."""
        qf = QualityFilter()
        score = qf._score_coherence(GOOD_TITLE, GOOD_BODY)
        assert score >= 0.5, f"Coherence should be ≥ 0.5 for matching title/body, got {score}"

    def test_mismatched_title_body_scores_low_coherence(self):
        """AC1: Title completely unrelated to body → low coherence score."""
        qf = QualityFilter()
        score = qf._score_coherence(
            "Aliens land on Mars and greet scientists",
            GOOD_BODY,
        )
        assert score < 0.5, f"Mismatched title/body coherence should be < 0.5, got {score}"


# ---------------------------------------------------------------------------
# AC2 - Clickbait articles rejected
# ---------------------------------------------------------------------------


class TestClickbaitRejection:
    def test_clickbait_title_is_detected(self):
        """AC2: _score_clickbait returns 0.0 for a clickbait title."""
        qf = QualityFilter()
        assert qf._score_clickbait(CLICKBAIT_TITLE) == 0.0

    def test_clean_title_passes_clickbait_check(self):
        """AC2: _score_clickbait returns 1.0 for a factual title."""
        qf = QualityFilter()
        assert qf._score_clickbait(GOOD_TITLE) == 1.0

    def test_clickbait_article_is_rejected(self):
        """AC2: Article with clickbait title is filtered out."""
        qf = QualityFilter()
        result = qf.filter(make_good_article(title=CLICKBAIT_TITLE))
        assert not result.passed
        assert "clickbait" in result.rejection_reason.lower()

    def test_good_article_passes_clickbait_filter(self):
        """AC2: Non-clickbait article with good body is not rejected for clickbait."""
        qf = QualityFilter()
        result = qf.filter(make_good_article())
        # May fail for other reasons, but not clickbait
        if not result.passed:
            assert "clickbait" not in result.rejection_reason.lower()

    def test_exclamation_marks_detected_as_clickbait(self):
        """AC2: Multiple exclamation marks in title trigger clickbait detection."""
        qf = QualityFilter()
        assert qf._score_clickbait("Breaking news!!!") == 0.0

    def test_shocking_keyword_detected(self):
        """AC2: 'shocking' keyword in title is flagged."""
        qf = QualityFilter()
        assert qf._score_clickbait("Shocking revelations about the budget") == 0.0


# ---------------------------------------------------------------------------
# AC3 - Articles with body < 100 words rejected as incomplete
# ---------------------------------------------------------------------------


class TestIncompleteRejection:
    def test_body_under_100_words_rejected(self):
        """AC3: Article with body < 100 words is rejected as incomplete."""
        qf = QualityFilter()
        short = make_good_article(body=SHORT_BODY)
        result = qf.filter(short)
        assert not result.passed
        assert "incomplete" in result.rejection_reason.lower() or "words" in result.rejection_reason.lower()

    def test_exactly_min_words_boundary(self):
        """AC3: Body with exactly MIN_BODY_WORDS words passes the word count check."""
        qf = QualityFilter()
        # Build a body with exactly MIN_BODY_WORDS meaningful words (5 sentences, 20 words each)
        sentence = "India defeated Australia in the cricket match at Mumbai's Wankhede Stadium on Saturday evening. "
        body = sentence * 8  # ~128 words
        word_count = len(body.split())
        assert word_count >= MIN_BODY_WORDS, f"Test body has only {word_count} words"
        score = qf._score_completeness(body)
        assert score == 1.0

    def test_99_word_body_scores_zero_completeness(self):
        """AC3: Body with 99 words gets 0.0 completeness score."""
        qf = QualityFilter()
        body_99 = " ".join(["word"] * 99)
        assert qf._score_completeness(body_99) == 0.0

    def test_100_word_body_with_sentences_passes_completeness(self):
        """AC3: Body with >= 100 words and >= 3 sentences passes completeness."""
        qf = QualityFilter()
        body = (
            "India defeated Australia by five wickets on Saturday at Mumbai Wankhede Stadium. "
            "Captain Rohit Sharma scored 87 runs off 64 balls to guide the team to victory. "
            "Jasprit Bumrah claimed three wickets in the middle overs sealing the series win. "
            "India won the series three to one and qualified for the World Cup final next month. "
            "The victory was celebrated across the country with fans cheering the national team. "
            "Bumrah was named player of the match for his outstanding bowling performance overall. "
            "The team management expressed immense satisfaction with how all players executed the game plan "
            "under intense pressure during the crucial final overs of the high-stakes encounter."
        )
        assert len(body.split()) >= 100
        assert qf._score_completeness(body) == 1.0


# ---------------------------------------------------------------------------
# AC4 - Sponsored content detected and filtered
# ---------------------------------------------------------------------------


class TestSponsoredContentFilter:
    def test_sponsored_keyword_detected(self):
        """AC4: 'sponsored' keyword triggers sponsored filter."""
        qf = QualityFilter()
        assert qf._score_sponsored(SPONSORED_BODY, GOOD_TITLE) == 0.0

    def test_advertorial_keyword_detected(self):
        """AC4: 'advertorial' keyword triggers sponsored filter."""
        qf = QualityFilter()
        body = "Advertorial: " + GOOD_BODY
        assert qf._score_sponsored(body, GOOD_TITLE) == 0.0

    def test_partner_content_detected(self):
        """AC4: 'partner content' phrase triggers sponsored filter."""
        qf = QualityFilter()
        body = "Partner content. " + GOOD_BODY
        assert qf._score_sponsored(body, GOOD_TITLE) == 0.0

    def test_clean_article_passes_sponsored_check(self):
        """AC4: Article with no sponsored markers scores 1.0 on sponsored dimension."""
        qf = QualityFilter()
        assert qf._score_sponsored(GOOD_BODY, GOOD_TITLE) == 1.0

    def test_sponsored_article_is_rejected(self):
        """AC4: filter() rejects sponsored content with appropriate reason."""
        qf = QualityFilter()
        result = qf.filter(make_good_article(body=SPONSORED_BODY))
        assert not result.passed
        assert "sponsor" in result.rejection_reason.lower() or "advertorial" in result.rejection_reason.lower()

    def test_press_release_detected_and_rejected(self):
        """AC4: Press release boilerplate detected and article rejected."""
        qf = QualityFilter()
        result = qf.filter(make_good_article(body=PR_BODY))
        assert not result.passed
        assert "press release" in result.rejection_reason.lower() or "pr" in result.rejection_reason.lower()


# ---------------------------------------------------------------------------
# AC5 - Quality threshold is configurable (default 0.4)
# ---------------------------------------------------------------------------


class TestConfigurableThreshold:
    def test_default_threshold_is_0_4(self):
        """AC5: Default threshold is 0.4."""
        qf = QualityFilter()
        assert qf.threshold == 0.4

    def test_custom_threshold_respected(self):
        """AC5: Custom threshold changes pass/fail boundary."""
        qf_strict = QualityFilter(threshold=0.9)
        qf_lenient = QualityFilter(threshold=0.1)

        article = make_good_article()
        score = qf_strict.score(article)

        # With strict threshold (0.9), a moderately good article may fail
        result_strict = qf_strict.filter(article)
        result_lenient = qf_lenient.filter(article)

        # Lenient should be at least as permissive as strict
        if result_strict.passed:
            assert result_lenient.passed
        # If score is between 0.1 and 0.9, strict fails and lenient passes
        if 0.1 <= score < 0.9:
            assert not result_strict.passed
            assert result_lenient.passed

    def test_threshold_0_passes_everything_above_hard_checks(self):
        """AC5: threshold=0.0 passes any article that clears hard-fail checks."""
        qf = QualityFilter(threshold=0.0)
        result = qf.filter(make_good_article())
        assert result.passed

    def test_threshold_1_fails_almost_everything(self):
        """AC5: threshold=1.0 rejects articles that score below perfect."""
        qf = QualityFilter(threshold=1.0)
        result = qf.filter(make_good_article())
        # A real article can't score exactly 1.0 across all dimensions
        assert not result.passed or qf.score(make_good_article()) == 1.0


# ---------------------------------------------------------------------------
# AC6 - Rejected articles logged with article_id, source, rejection_reason, score
# ---------------------------------------------------------------------------


class TestRejectionLogging:
    def test_rejection_logged_with_all_required_fields(self):
        """AC6: Rejected article appears in log with article_id, source, reason, score."""
        qf = QualityFilter()
        qf.clear_log()

        article = make_good_article(
            article_id="art-999",
            source="NDTV",
            title=CLICKBAIT_TITLE,
        )
        result = qf.filter(article)
        assert not result.passed

        log = qf.get_rejection_log()
        assert len(log) == 1
        record = log[0]

        assert isinstance(record, RejectionRecord)
        assert record.article_id == "art-999"
        assert record.source == "NDTV"
        assert record.rejection_reason is not None and len(record.rejection_reason) > 0
        assert isinstance(record.score, float)

    def test_passed_articles_not_logged(self):
        """AC6: Articles that pass filter are NOT added to rejection log."""
        qf = QualityFilter(threshold=0.0)  # lenient so good article passes
        qf.clear_log()
        qf.filter(make_good_article())
        assert len(qf.get_rejection_log()) == 0

    def test_multiple_rejections_all_logged(self):
        """AC6: Multiple rejected articles all appear in log."""
        qf = QualityFilter()
        qf.clear_log()

        articles = [make_good_article(article_id=f"art-{i}", title=CLICKBAIT_TITLE) for i in range(5)]
        for art in articles:
            qf.filter(art)

        assert len(qf.get_rejection_log()) == 5

    def test_clear_log_empties_rejection_log(self):
        """AC6: clear_log() resets the rejection log to empty."""
        qf = QualityFilter()
        qf.filter(make_good_article(title=CLICKBAIT_TITLE))
        assert len(qf.get_rejection_log()) > 0
        qf.clear_log()
        assert qf.get_rejection_log() == []

    def test_rejection_score_matches_score_method(self):
        """AC6: Score in rejection log matches what score() returns."""
        qf = QualityFilter()
        qf.clear_log()
        article = make_good_article(title=CLICKBAIT_TITLE)
        expected_score = qf.score(article)
        qf.filter(article)
        record = qf.get_rejection_log()[0]
        assert record.score == expected_score


# ---------------------------------------------------------------------------
# AC7 - GET /api/admin/rejected-articles returns rejections with reasons
# ---------------------------------------------------------------------------


class TestAdminAPI:
    def setup_method(self):
        """Reset the admin singleton filter before each test."""
        import src.api.admin as admin_mod

        admin_mod._filter_instance = None

    def test_rejected_articles_endpoint_returns_200(self):
        """AC7: GET /api/admin/rejected-articles returns HTTP 200."""
        client = TestClient(app)
        response = client.get("/api/admin/rejected-articles")
        assert response.status_code == 200

    def test_empty_log_returns_zero_rejections(self):
        """AC7: With no rejections, endpoint returns empty list."""
        client = TestClient(app)
        data = client.get("/api/admin/rejected-articles").json()
        assert data["total"] == 0
        assert data["rejections"] == []

    def test_rejection_appears_in_api_response(self):
        """AC7: After a rejection, it appears in /api/admin/rejected-articles."""
        import src.api.admin as admin_mod

        # Inject a pre-populated filter
        qf = QualityFilter()
        qf.filter(
            make_good_article(
                article_id="api-test-001",
                source="Times of India",
                title=CLICKBAIT_TITLE,
            )
        )
        admin_mod._filter_instance = qf

        client = TestClient(app)
        data = client.get("/api/admin/rejected-articles").json()

        assert data["total"] >= 1
        assert data["returned"] >= 1
        record = data["rejections"][0]
        assert "article_id" in record
        assert "source" in record
        assert "rejection_reason" in record
        assert "score" in record
        assert record["rejection_reason"] is not None

    def test_limit_parameter_respected(self):
        """AC7: limit query param caps the number of returned rejections."""
        import src.api.admin as admin_mod

        qf = QualityFilter()
        for i in range(10):
            qf.filter(make_good_article(article_id=f"art-{i}", title=CLICKBAIT_TITLE))
        admin_mod._filter_instance = qf

        client = TestClient(app)
        data = client.get("/api/admin/rejected-articles?limit=3").json()
        assert data["returned"] == 3
        assert len(data["rejections"]) == 3

    def test_response_schema_has_required_keys(self):
        """AC7: API response always contains total, returned, rejections keys."""
        client = TestClient(app)
        data = client.get("/api/admin/rejected-articles").json()
        assert "total" in data
        assert "returned" in data
        assert "rejections" in data


# ---------------------------------------------------------------------------
# AC8 - Filter processes 100 articles in < 10 seconds
# ---------------------------------------------------------------------------


class TestBatchPerformance:
    def test_100_articles_under_10_seconds(self):
        """AC8: batch_filter() processes 100 articles in < 10 seconds."""
        qf = QualityFilter()
        articles = [make_good_article(article_id=f"art-{i}", source=f"source-{i}") for i in range(100)]

        start = time.time()
        passed, rejections = qf.batch_filter(articles)
        elapsed = time.time() - start

        assert elapsed < 10.0, f"batch_filter(100) took {elapsed:.2f}s, limit is 10s"
        assert len(passed) + len(rejections) == 100

    def test_batch_filter_returns_two_lists(self):
        """AC8: batch_filter() returns (passed_articles, rejection_records)."""
        qf = QualityFilter()
        good = [make_good_article(article_id="g1")]
        bad = [make_good_article(article_id="b1", title=CLICKBAIT_TITLE)]
        passed, rejections = qf.batch_filter(good + bad)

        assert isinstance(passed, list)
        assert isinstance(rejections, list)

    def test_batch_separates_good_and_bad(self):
        """AC8: batch_filter correctly separates passing and rejected articles."""
        qf = QualityFilter(threshold=0.0)  # only hard checks apply
        good = make_good_article(article_id="good")
        bad = make_good_article(article_id="bad", title=CLICKBAIT_TITLE)

        passed, rejections = qf.batch_filter([good, bad])

        passed_ids = [a.get("article_id") for a in passed]
        rejected_ids = [r.article_id for r in rejections]

        assert "good" in passed_ids
        assert "bad" in rejected_ids
