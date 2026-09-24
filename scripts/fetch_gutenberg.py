#!/usr/bin/env python3
"""Downloads the real Project Gutenberg books used by the opt-in
`@pytest.mark.gutenberg` layer of the integration suite
(tests/integration/gutenberg.py) into a local, gitignored cache.

Run manually before `pytest -m gutenberg`:

    python scripts/fetch_gutenberg.py

Idempotent — skips any book already present in the cache. Network-only
script; nothing here is imported by the test suite itself, so a machine
that never runs this simply never gets `gutenberg`-marked tests collected
with real files (they skip cleanly — see gutenberg.py).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "tests" / "integration" / ".gutenberg_cache"

# (cache key, Project Gutenberg ebook id, title) — chosen for structural
# variety: some famously clean TOCs, some very long, some very short, at
# least one (Pride and Prejudice) with the documented messy-TOC-label
# pathology CLAUDE.md already discusses.
BOOKS: tuple[tuple[str, int, str], ...] = (
    ("pride_and_prejudice", 1342, "Pride and Prejudice"),
    ("tale_of_two_cities", 98, "A Tale of Two Cities"),
    ("sherlock_holmes", 1661, "The Adventures of Sherlock Holmes"),
    ("frankenstein", 84, "Frankenstein"),
    ("moby_dick", 2701, "Moby Dick"),
    ("yellow_wallpaper", 1952, "The Yellow Wallpaper"),
    ("alice_in_wonderland", 11, "Alice's Adventures in Wonderland"),
    ("jekyll_and_hyde", 43, "The Strange Case of Dr Jekyll and Mr Hyde"),
)

URL_TEMPLATE = "https://www.gutenberg.org/ebooks/{id}.epub.noimages"


def fetch_all(cache_dir: Path = CACHE_DIR, *, force: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, gid, title in BOOKS:
        dest = cache_dir / f"{key}.epub"
        if dest.exists() and not force:
            print(f"[skip]  {title} (already cached at {dest})")
            continue
        url = URL_TEMPLATE.format(id=gid)
        print(f"[fetch] {title} <- {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except Exception as e:  # noqa: BLE001 - reported, not raised, so one failure doesn't abort the batch
            print(f"[FAIL]  {title}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        dest.write_bytes(data)
        print(f"[ok]    {title}: {len(data) / 1024:.0f} KB -> {dest}")


if __name__ == "__main__":
    fetch_all(force="--force" in sys.argv)
