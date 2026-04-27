"""
share.py — Generate deep-links to social platform post composers.
"""

import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

# LinkedIn allows up to 3000 chars in a post; we stay comfortably under
_LINKEDIN_MAX_CHARS = 2800
_TWITTER_MAX_CHARS  = 280

_LINKEDIN_SHARE_URL = "https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fresearchrag.local&summary={text}"
_TWITTER_COMPOSE_URL = "https://twitter.com/intent/tweet?text={text}"

_ELLIPSIS = "…"


def _join_hashtags(hashtags: list[str]) -> str:
    """Format hashtags, adding # prefix if missing."""
    formatted = []
    for tag in hashtags:
        tag = str(tag).strip()
        if tag and not tag.startswith("#"):
            tag = f"#{tag}"
        if tag:
            formatted.append(tag)
    return " ".join(formatted)


def linkedin_deeplink(caption: str, hashtags: list[str]) -> str:
    """
    Build a LinkedIn share deep-link with pre-filled post text.

    Combines caption and hashtags, truncates to LinkedIn's limit,
    URL-encodes the result, and returns the share URL.
    """
    hashtag_str = _join_hashtags(hashtags)
    full_text   = f"{caption}\n\n{hashtag_str}".strip() if hashtag_str else caption.strip()

    # Truncate to stay within LinkedIn's character limit.
    # FIX S2: Subtract len(_ELLIPSIS) from available so that appending "…"
    # does not push the final string over _LINKEDIN_MAX_CHARS.
    if len(full_text) > _LINKEDIN_MAX_CHARS:
        available = _LINKEDIN_MAX_CHARS - len(hashtag_str) - 2 - len(_ELLIPSIS)
        if available > 50:
            truncated_caption = caption[:available].rstrip() + _ELLIPSIS
            full_text = f"{truncated_caption}\n\n{hashtag_str}".strip()
        else:
            full_text = full_text[:_LINKEDIN_MAX_CHARS]
        logger.debug("LinkedIn text truncated to %d chars", len(full_text))

    encoded = quote(full_text, safe="")
    url = _LINKEDIN_SHARE_URL.format(text=encoded)
    logger.info("LinkedIn deep-link generated (%d chars)", len(full_text))
    return url


def twitter_deeplink(text: str, hashtags: list[str]) -> str:
    """
    Build a Twitter/X compose deep-link with pre-filled tweet text.

    Combines text and hashtags, truncates to 280 characters,
    URL-encodes the result, and returns the compose URL.
    """
    hashtag_str = _join_hashtags(hashtags)
    full_text   = f"{text} {hashtag_str}".strip() if hashtag_str else text.strip()

    # Truncate to stay within Twitter's character limit.
    # FIX S1: Subtract len(_ELLIPSIS) from available so that appending "…"
    # does not push the final string over _TWITTER_MAX_CHARS.
    if len(full_text) > _TWITTER_MAX_CHARS:
        available = _TWITTER_MAX_CHARS - len(hashtag_str) - 1 - len(_ELLIPSIS)
        if available > 20:
            truncated_text = text[:available].rstrip() + _ELLIPSIS
            full_text = f"{truncated_text} {hashtag_str}".strip()
        else:
            full_text = full_text[:_TWITTER_MAX_CHARS]
        logger.debug("Twitter text truncated to %d chars", len(full_text))

    encoded = quote(full_text, safe="")
    url = _TWITTER_COMPOSE_URL.format(text=encoded)
    logger.info("Twitter deep-link generated (%d chars)", len(full_text))
    return url