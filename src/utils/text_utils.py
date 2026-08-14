"""NewsSnap AI - Text Processing Utilities."""

import re

from bs4 import BeautifulSoup


def clean_html(html_str: str) -> str:
    """Removes script and style tags from HTML string and returns the cleaned HTML."""
    if not html_str:
        return ""
    soup = BeautifulSoup(html_str, "html.parser")
    # Remove script and style tags
    for element in soup(["script", "style"]):
        element.extract()
    return str(soup)


def extract_text(html_str: str) -> str:
    """Extracts plain text from an HTML string, removing all tags."""
    if not html_str:
        return ""
    # Use BeautifulSoup to get text
    soup = BeautifulSoup(html_str, "html.parser")
    # Remove script and style tags
    for element in soup(["script", "style"]):
        element.extract()
    text = soup.get_text(separator=" ")
    return text


def normalize_whitespace(text: str) -> str:
    """Replaces multiple whitespaces and newlines with a single space and trims."""
    if not text:
        return ""
    # Replace any whitespace character sequence with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()
