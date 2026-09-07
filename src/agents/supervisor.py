"""NewsSnap AI - Supervisor Agent (Issue 11 pipeline orchestration).

Orchestrates the full article processing pipeline:
  1. Deduplication (DedupAgent)
  2. Story Clustering (StoryClusterer)
  3. Summarization (SummarizerAgent)
"""

import logging
from typing import Any, Dict, List, Optional

from src.agents.dedup_agent import DedupAgent
from src.agents.story_clusterer import StoryClusterer
from src.agents.summarizer_agent import SummarizerAgent

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Orchestrates the full article processing pipeline."""

    def __init__(
        self,
        dedup_agent: Optional[DedupAgent] = None,
        story_clusterer: Optional[StoryClusterer] = None,
        summarizer: Optional[SummarizerAgent] = None,
    ):
        self.dedup_agent = dedup_agent or DedupAgent()
        self.story_clusterer = story_clusterer or StoryClusterer()
        self.summarizer = summarizer or SummarizerAgent()

    def process(
        self,
        articles: List[Dict[str, Any]],
        db=None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on a batch of raw scraped articles.

        Steps:
            1. Deduplication — remove semantic duplicates.
            2. Story Clustering — group related unique articles into stories.
            3. Summarization — generate 60-80 word summaries for each unique article.

        Returns:
            dict with keys: unique, duplicates, stories, summaries.
        """
        # Step 1: Deduplicate
        logger.info(f"Supervisor: deduplicating {len(articles)} articles...")
        dedup_result = self.dedup_agent.check_duplicates(articles, db=db)

        # Step 2: Cluster unique articles into stories
        logger.info(f"Supervisor: clustering {len(dedup_result.unique)} unique articles into stories...")
        stories = self.story_clusterer.cluster_articles(dedup_result.unique, db=db)

        # Step 3: Summarize unique articles
        logger.info("Supervisor: summarizing unique articles...")
        summaries = self.summarizer.batch_summarize(dedup_result.unique)

        logger.info(
            f"Supervisor pipeline complete: "
            f"{len(dedup_result.unique)} unique, "
            f"{len(dedup_result.duplicates)} duplicates, "
            f"{len(stories)} stories, "
            f"{len(summaries)} summaries."
        )

        return {
            "unique": dedup_result.unique,
            "duplicates": dedup_result.duplicates,
            "stories": stories,
            "summaries": summaries,
        }
