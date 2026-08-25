"""Shared test isolation.

Two hazards this file removes, both found by mutation testing:

1. `_fetch_youtube_page` is `lru_cache`d, and the cache outlived individual
   tests. Three tests in `test_coverage_boost.py` shared the video id "abc",
   so the second and third read the first one's cached HTML and never
   exercised their own path — including one that claimed to test network
   failure and provably did not (removing the failure guard entirely left it
   green).

2. The suite made 8 real requests to youtube.com. They were invisible because
   `_fetch_youtube_page` swallows every exception, so an offline machine
   silently took a different branch than an online one — which also made the
   coverage number depend on network state.
"""

from __future__ import annotations

import pytest

from youtube_gemini_processor import cli


@pytest.fixture(autouse=True)
def _clear_youtube_page_cache():
    """Give every test a cold page cache in both directions."""
    cli._fetch_youtube_page.cache_clear()
    yield
    cli._fetch_youtube_page.cache_clear()


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """Fail any unmocked outbound request the way an offline machine would.

    Raising OSError (rather than something exotic) keeps the code path
    identical to a genuinely offline run, so the branch coverage of the
    fetch helper's error handling is deterministic. A test that wants to
    exercise a real fetch path patches `urlopen` itself, which wins over
    this fixture.
    """

    def _refuse(*args, **kwargs):
        raise OSError("outbound network is blocked in tests; patch urlopen instead")

    monkeypatch.setattr("urllib.request.urlopen", _refuse)
