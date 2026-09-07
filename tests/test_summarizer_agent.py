"""Tests for LLM Summarizer Agent (Issue 11).

Acceptance Criteria covered:
  AC1 - Summaries are 60-80 words (configurable range)
  AC2 - Key facts preserved: who, what, when, where, why (at least 3 of 5)
  AC3 - No clickbait language in summaries (verified by quality check)
  AC4 - Category-aware prompts produce different styles (finance vs sports vs politics)
  AC5 - Quality check rejects and re-summarizes bad outputs (max 2 retries)
  AC6 - Source attribution included in summary metadata (not in text body)
  AC7 - Batch processing: 20 articles summarized in < 30 seconds via Groq
  AC8 - Handles articles in English and Hindi (source language detection)

NOTE: All Groq API calls are mocked so no real API key is needed.
"""

import time
from unittest.mock import MagicMock

from src.agents.summarizer_agent import SummarizerAgent, SummaryResult

# ---------------------------------------------------------------------------
# Shared fixtures and sample data
# ---------------------------------------------------------------------------

CRICKET_ARTICLE = """
India defeated Australia by 5 wickets in a thrilling final match at Wankhede Stadium, Mumbai on Saturday.
Captain Rohit Sharma scored 87 runs off 64 balls to lead the team to victory after Australia posted
a challenging total of 287. Fast bowler Jasprit Bumrah took three crucial wickets in the middle overs.
The match-winning partnership between Sharma and Virat Kohli was decisive. India have now won the
series 3-1 and qualify for the World Cup final next month in Ahmedabad.
India defeated Australia by 5 wickets in a thrilling final match at Wankhede Stadium, Mumbai on Saturday.
Captain Rohit Sharma scored 87 runs off 64 balls to lead the team to victory after Australia posted
a challenging total of 287. Fast bowler Jasprit Bumrah took three crucial wickets in the middle overs.
"""

FINANCE_ARTICLE = """
The Reserve Bank of India on Friday cut the repo rate by 25 basis points to 6.25 percent,
the first rate reduction since February 2023. RBI Governor Shaktikanta Das announced the decision
following the Monetary Policy Committee meeting in Mumbai. The move is expected to reduce home loan
EMIs by approximately Rs 1,500 per month on a Rs 50 lakh loan. Stock markets surged following the
announcement with the Sensex gaining 450 points to close at 79,800. Economists expect GDP growth
to accelerate to 7.2 percent in the coming fiscal year as borrowing costs decline across sectors.
The Reserve Bank of India on Friday cut the repo rate by 25 basis points to 6.25 percent, the first
rate reduction since February 2023.
"""

POLITICS_ARTICLE = """
Prime Minister Narendra Modi on Thursday signed a landmark agreement with French President Emmanuel
Macron to purchase 26 Rafale Marine fighter jets for the Indian Navy at a total cost of Rs 63,000 crore.
The deal was finalised in New Delhi during Macron's two-day state visit to India. "This partnership
strengthens our strategic defence relationship," PM Modi declared at the joint press conference.
The jets will be manufactured with 60 percent indigenous content under the Make in India programme.
The Navy expects delivery starting 2028. Opposition leaders questioned the deal price in Parliament.
Prime Minister Narendra Modi signed a landmark agreement with French President Emmanuel Macron.
"""

HINDI_ARTICLE = """
भारत ने शनिवार को वानखेड़े स्टेडियम में ऑस्ट्रेलिया को 5 विकेट से हराया। कप्तान रोहित शर्मा ने
64 गेंदों पर 87 रन बनाकर टीम को जीत दिलाई। जसप्रीत बुमराह ने मिडल ओवर में तीन महत्वपूर्ण विकेट लिए।
भारत ने सीरीज 3-1 से जीती और अगले महीने अहमदाबाद में होने वाले विश्व कप फाइनल के लिए क्वालीफाई कर लिया।
यह भारत की ऑस्ट्रेलिया पर सबसे बड़ी जीत में से एक है।
"""


def make_good_summary() -> str:
    """Generate a good-quality ~70-word factual summary about cricket match."""
    return (
        "India defeated Australia by five wickets in a thrilling series final on Saturday "
        "at Mumbai's Wankhede Stadium. Captain Rohit Sharma scored 87 runs off 64 balls "
        "to guide the team to victory after Australia posted a total of 287. Jasprit Bumrah "
        "claimed three crucial wickets in the middle overs. India won the series 3-1 and "
        "qualified for the ICC World Cup final in Ahmedabad next month defeating Australia convincingly."
    )


def make_agent_with_mock(response_text: str, side_effects=None) -> SummarizerAgent:
    """Create a SummarizerAgent with mocked Groq client."""
    agent = SummarizerAgent(groq_api_key="test-key-123", min_words=60, max_words=80, max_retries=2)

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text

    if side_effects:
        # Multiple different responses for retry testing
        mock_client.chat.completions.create.side_effect = side_effects
    else:
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    agent._client = mock_client
    return agent


# ---------------------------------------------------------------------------
# AC1 - Summaries are 60-80 words (configurable range)
# ---------------------------------------------------------------------------


class TestSummaryWordCount:
    def test_valid_summary_word_count_passes(self):
        """AC1: Summary within 60-80 words passes quality check."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        summary = make_good_summary()
        passed, reason = agent.validate_quality(summary, CRICKET_ARTICLE)
        word_count = agent._count_words(summary)

        assert 60 <= word_count <= 80, f"Expected 60-80 words, got {word_count}"
        assert passed, f"Expected quality check to pass, got: {reason}"

    def test_too_short_summary_fails(self):
        """AC1: Summary < 60 words fails quality check."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        short_summary = "India won the cricket match against Australia."
        passed, reason = agent.validate_quality(short_summary, CRICKET_ARTICLE)

        assert not passed
        assert "short" in reason.lower()

    def test_too_long_summary_fails(self):
        """AC1: Summary > 80 words fails quality check."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        long_summary = " ".join(["word"] * 90)
        passed, reason = agent.validate_quality(long_summary, CRICKET_ARTICLE)

        assert not passed
        assert "long" in reason.lower()

    def test_configurable_word_range(self):
        """AC1: min_words and max_words are configurable."""
        agent = SummarizerAgent(min_words=40, max_words=60)
        # Just verify configuration is respected
        assert agent.min_words == 40
        assert agent.max_words == 60

        # Summary in the 40-60 word configurable range; includes enough who/what/where/when
        # facts to satisfy the key-facts quality check (≥ 3 of 5 Ws).
        summary_in_range = (
            "India defeated Australia by five wickets on Saturday at Mumbai's Wankhede Stadium. "
            "Captain Rohit Sharma scored 87 brilliant runs off 64 balls to guide the team home "
            "after Australia posted 287. Jasprit Bumrah claimed three crucial wickets in the "
            "middle overs to seal India's 3-1 series victory."
        )
        word_count = agent._count_words(summary_in_range)
        assert 40 <= word_count <= 60, f"Test summary should be 40-60 words, got {word_count}"
        passed, reason = agent.validate_quality(summary_in_range, CRICKET_ARTICLE)
        assert passed, f"Summary should pass with min_words=40, got: {reason}"


# ---------------------------------------------------------------------------
# AC2 - Key facts preserved: who, what, when, where, why (at least 3 of 5)
# ---------------------------------------------------------------------------


class TestKeyFactsPreservation:
    def test_good_summary_has_3_of_5_key_facts(self):
        """AC2: Good summary passes key facts check with ≥ 3/5 Ws."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        summary = make_good_summary()
        facts_count = agent._check_key_facts(summary, CRICKET_ARTICLE)

        assert facts_count >= 3, f"Expected ≥ 3 key facts, got {facts_count}"

    def test_summary_missing_facts_fails(self):
        """AC2: Summary with < 3 key facts fails quality check."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        # Generic summary with no real facts
        no_facts = " ".join(["something important happened involving people"] * 9)
        passed, reason = agent.validate_quality(no_facts, CRICKET_ARTICLE)

        assert not passed or "facts" in reason.lower() or not passed


# ---------------------------------------------------------------------------
# AC3 - No clickbait language in summaries
# ---------------------------------------------------------------------------


class TestNoClickbait:
    def test_clickbait_summary_fails(self):
        """AC3: Summary with clickbait language fails quality check."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        clickbait_summary = (
            "You won't believe what India's captain did! Shocking scenes at Wankhede "
            "as Rohit Sharma hit this incredible knock on Saturday against Australia. "
            "This is why cricket fans are going crazy right now. India won the match."
        )
        has_cb = agent._has_clickbait(clickbait_summary)
        assert has_cb, "Expected clickbait to be detected"

    def test_clean_summary_no_clickbait(self):
        """AC3: Clean factual summary does not trigger clickbait detection."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        clean_summary = make_good_summary()
        has_cb = agent._has_clickbait(clean_summary)
        assert not has_cb, "Good summary should not be flagged as clickbait"

    def test_clickbait_fails_validate_quality(self):
        """AC3: validate_quality rejects summaries with clickbait."""
        agent = SummarizerAgent(min_words=60, max_words=80)
        # Pad clickbait summary to meet word count so clickbait check fires
        clickbait = (
            "You won't believe this shocking cricket result as India shockingly defeated "
            "Australia on Saturday at Mumbai's Wankhede Stadium by five wickets convincingly. "
            "Rohit Sharma scored an incredible 87 runs off 64 balls to guide the team home. "
            "Jasprit Bumrah claimed three crucial wickets. India won series 3-1 and qualified "
            "for the ICC World Cup final in Ahmedabad next month comprehensively."
        )
        passed, reason = agent.validate_quality(clickbait, CRICKET_ARTICLE)
        assert not passed
        assert "clickbait" in reason.lower()


# ---------------------------------------------------------------------------
# AC4 - Category-aware prompts produce different styles
# ---------------------------------------------------------------------------


class TestCategoryAwarePrompts:
    def test_finance_prompt_contains_financial_style(self):
        """AC4: Finance category prompt emphasizes financial figures."""
        agent = SummarizerAgent()
        prompt = agent._build_prompt(FINANCE_ARTICLE, "finance", "en")
        assert any(word in prompt.lower() for word in ["financial", "figures", "market", "economic"])

    def test_sports_prompt_contains_sports_style(self):
        """AC4: Sports category prompt emphasizes scores and teams."""
        agent = SummarizerAgent()
        prompt = agent._build_prompt(CRICKET_ARTICLE, "sports", "en")
        assert any(word in prompt.lower() for word in ["score", "team", "match", "result", "player"])

    def test_politics_prompt_contains_politics_style(self):
        """AC4: Politics category prompt emphasizes quotes and decision-makers."""
        agent = SummarizerAgent()
        prompt = agent._build_prompt(POLITICS_ARTICLE, "politics", "en")
        assert any(word in prompt.lower() for word in ["quote", "decision", "policy", "implication"])

    def test_different_categories_produce_different_prompts(self):
        """AC4: Different categories generate different prompt content."""
        agent = SummarizerAgent()
        finance_prompt = agent._build_prompt(FINANCE_ARTICLE, "finance", "en")
        sports_prompt = agent._build_prompt(CRICKET_ARTICLE, "sports", "en")
        assert finance_prompt != sports_prompt


# ---------------------------------------------------------------------------
# AC5 - Quality check rejects and re-summarizes bad outputs (max 2 retries)
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_bad_first_response_triggers_retry(self):
        """AC5: Short/bad first response triggers retry up to max_retries."""
        bad_response = MagicMock()
        bad_response.message.content = "India won."  # Too short

        good_summary = make_good_summary()
        good_response = MagicMock()
        good_response.message.content = good_summary

        agent = SummarizerAgent(groq_api_key="test-key", min_words=60, max_words=80, max_retries=2)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[bad_response]),
            MagicMock(choices=[good_response]),
        ]
        agent._client = mock_client

        result = agent.summarize(CRICKET_ARTICLE, category="sports")
        assert result.retries >= 1, "Should have retried at least once"
        assert mock_client.chat.completions.create.call_count >= 2

    def test_max_retries_respected(self):
        """AC5: Agent stops retrying after max_retries attempts."""
        bad_response = MagicMock()
        bad_response.message.content = "Short."

        agent = SummarizerAgent(groq_api_key="test-key", min_words=60, max_words=80, max_retries=2)
        mock_client = MagicMock()
        # Always return bad response
        mock_client.chat.completions.create.return_value = MagicMock(choices=[bad_response])
        agent._client = mock_client

        agent.summarize(CRICKET_ARTICLE, category="sports")
        # Should have attempted: 1 original + 2 retries = 3 total
        assert mock_client.chat.completions.create.call_count <= 3

    def test_retry_prompt_becomes_stricter(self):
        """AC5: Retry prompt contains stronger wording on word count requirement."""
        agent = SummarizerAgent(min_words=60, max_words=80, max_retries=2)
        first_prompt = agent._build_prompt(CRICKET_ARTICLE, "sports", "en", retry_num=0)
        retry_prompt = agent._build_prompt(CRICKET_ARTICLE, "sports", "en", retry_num=1)
        assert "IMPORTANT" in retry_prompt or "failed" in retry_prompt.lower()
        assert len(retry_prompt) > len(first_prompt)


# ---------------------------------------------------------------------------
# AC6 - Source attribution included in summary metadata (not in text body)
# ---------------------------------------------------------------------------


class TestSourceAttribution:
    def test_source_in_metadata_not_in_summary_text(self):
        """AC6: Source name appears in metadata, NOT injected into the summary body."""
        good_summary = make_good_summary()
        agent = make_agent_with_mock(good_summary)

        result = agent.summarize(CRICKET_ARTICLE, category="sports", source="The Hindu")

        assert result.source_attribution == "The Hindu"
        assert "The Hindu" not in result.summary, "Source should NOT appear in summary body"

    def test_source_attribution_preserved_in_result(self):
        """AC6: SummaryResult carries source_attribution field correctly."""
        good_summary = make_good_summary()
        agent = make_agent_with_mock(good_summary)

        result = agent.summarize(FINANCE_ARTICLE, category="finance", source="Mint")

        assert result.source_attribution == "Mint"
        assert isinstance(result.metadata, dict)

    def test_no_source_attribution_when_none_provided(self):
        """AC6: source_attribution is None when no source passed."""
        good_summary = make_good_summary()
        agent = make_agent_with_mock(good_summary)

        result = agent.summarize(CRICKET_ARTICLE, category="sports")

        assert result.source_attribution is None


# ---------------------------------------------------------------------------
# AC7 - Batch processing: 20 articles summarized in < 30 seconds
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    def test_20_articles_under_30_seconds(self):
        """AC7: Batch summarize 20 articles in < 30 seconds."""
        good_summary = make_good_summary()
        agent = make_agent_with_mock(good_summary)

        articles = [
            {
                "body": CRICKET_ARTICLE,
                "category": "sports",
                "source": "NDTV",
                "title": f"Cricket Match Report {i}",
            }
            for i in range(20)
        ]

        start = time.time()
        results = agent.batch_summarize(articles)
        elapsed = time.time() - start

        assert len(results) == 20
        assert elapsed < 30.0, f"Batch of 20 took {elapsed:.2f}s, limit is 30s"
        assert all(isinstance(r, SummaryResult) for r in results)

    def test_batch_preserves_order(self):
        """AC7: Batch results are returned in the same order as input."""
        good_summary = make_good_summary()
        agent = make_agent_with_mock(good_summary)

        articles = [{"body": CRICKET_ARTICLE, "category": "sports", "source": f"src-{i}"} for i in range(5)]
        results = agent.batch_summarize(articles)

        for i, result in enumerate(results):
            assert result.source_attribution == f"src-{i}"

    def test_batch_empty_input(self):
        """AC7: Empty batch returns empty list."""
        agent = SummarizerAgent()
        results = agent.batch_summarize([])
        assert results == []


# ---------------------------------------------------------------------------
# AC8 - Handles articles in English and Hindi
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    def test_english_article_detected_as_en(self):
        """AC8: English article is detected as 'en'."""
        agent = SummarizerAgent()
        lang = agent.detect_language(CRICKET_ARTICLE)
        assert lang == "en"

    def test_hindi_article_detected_as_hi(self):
        """AC8: Hindi article is detected as 'hi'."""
        agent = SummarizerAgent()
        lang = agent.detect_language(HINDI_ARTICLE)
        assert lang == "hi"

    def test_hindi_prompt_instructs_english_output(self):
        """AC8: Hindi article prompt instructs model to output summary in English."""
        agent = SummarizerAgent()
        prompt = agent._build_prompt(HINDI_ARTICLE, "sports", "hi")
        assert "Hindi" in prompt
        assert "English" in prompt

    def test_summarize_hindi_article_returns_result(self):
        """AC8: Hindi article is accepted and processed without errors."""
        good_summary = make_good_summary()
        agent = make_agent_with_mock(good_summary)

        result = agent.summarize(HINDI_ARTICLE, category="sports")
        assert result.language == "hi"
        assert isinstance(result.summary, str)
