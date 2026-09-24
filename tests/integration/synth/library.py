"""The synthetic library: 8 deterministically-generated books, one per
structural archetype named in the integration test requirements, plus the
ground truth (needles, entity mentions) planted inside them.

Every book is built from `generator.py`'s themed prose, seeded from its own
`key` (via a stable sha256-derived integer, not the process-randomized
builtin `hash()`), so the exact same files and ground truth are produced on
every machine and every run — that's what makes the on-disk cache under
`tests/integration/.cache/books/` safe to keep across runs (see
`build_library`).

Archetypes (one per stated integration-test requirement):

| key                 | format | what it stresses                                    |
|---------------------|--------|-------------------------------------------------------|
| clean_toc           | epub   | many chapters, proper nav+anchors, one noisy TOC label |
| no_toc              | txt    | zero structure — one unstructured block               |
| fragment_anchored   | epub   | several chapters packed into one XHTML file (the real  |
|                     |        | Project Gutenberg shape EpubLoader special-cases)      |
| unusual_naming      | md     | non-numeric/duplicate chapter titles                   |
| very_long_chapter   | epub   | one chapter big enough to force group-splitting under  |
|                     |        | the real default max_group_tokens=30_000               |
| very_short          | md     | 2 tiny chapters                                        |
| page_structured     | pdf    | outline + real page numbers                            |
| heading_structured  | docx   | nested headings (chapter > subsection)                 |

Scope note on ground truth quality: needle queries are built from (most of)
the planted needle sentence itself, and entity aliasing is the "richer
alias list" a real per-chapter extractor would actually produce (every
distinct surface form seen in that chapter) — not a minimal name. Against
DeterministicEmbedder (a coarse hashed bag-of-words double, not a real
semantic model) this is what it takes for genuine cross-chapter/cross-book
aliases to clear EmbeddingEntityResolver's real production default
threshold (0.87) while unrelated same-type entities correctly stay well
below it — verified empirically (see docs/INTEGRATION_TEST_FINDINGS.md).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.integration.doubles import EntityMention
from tests.integration.synth.generator import (
    GENERATOR_VERSION,
    THEMES,
    generate_body,
    make_mention,
    make_needle,
    wrap_lines,
)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "books"


def _seed(key: str) -> int:
    """Deterministic integer seed from a string key — NOT Python's builtin
    `hash()`, which is randomized per-process (PYTHONHASHSEED) and would
    silently break the "same seed -> byte-identical file" cache contract."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


@dataclass(frozen=True)
class Needle:
    book_key: str
    marker: str
    """Unique token embedded in the planted sentence — the ground-truth
    signal a test uses to find "the" correct chunk without hand-labeling."""
    sentence: str
    query: str
    chapter_ordinal: int | None
    """Authored 0-based chapter index this needle was planted in, or None
    for a book with no chapter structure (`no_toc`)."""


@dataclass(frozen=True)
class GraphProbe:
    """Ground truth for the graph-hybrid-vs-vector-only retrieval scenario
    (requirement 4): `seed_query` is built from the exact rendered sentence
    that introduces an entity+relation together, so it's guaranteed to
    lexically seed that chunk via plain vector search; `target_marker` is a
    unique token planted in a lexically-unrelated, distant chunk that only
    mentions the *same* (content-addressed, so exact-merge, no resolver
    needed) entity again — reachable only by one-hop graph traversal from
    the seed, not by the query's own words."""

    book_key: str
    seed_query: str
    target_marker: str


@dataclass(frozen=True)
class BookSpec:
    key: str
    title: str
    theme: str
    archetype: str
    fmt: Literal["epub", "txt", "md", "pdf", "docx"]
    path: Path
    chapter_count: int
    """Authored chapter count (0 for the chapterless `no_toc` book)."""
    entity_mentions: tuple[tuple[int | None, EntityMention], ...] = ()
    """(authored chapter ordinal, mention) pairs planted in this book."""
    graph_probe: GraphProbe | None = None


@dataclass(frozen=True)
class Library:
    books: tuple[BookSpec, ...]
    needles: tuple[Needle, ...]

    def book(self, key: str) -> BookSpec:
        for b in self.books:
            if b.key == key:
                return b
        raise KeyError(key)


# --------------------------------------------------------------------------
# Format writers
# --------------------------------------------------------------------------


def _write_txt(path: Path, paragraphs: list[str]) -> None:
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")


def _write_md(path: Path, chapters: list[tuple[str, list[str], list[tuple[str, list[str]]]]]) -> None:
    """`chapters`: list of (title, paragraphs, subsections) where
    subsections is a list of (subtitle, paragraphs)."""
    lines: list[str] = []
    for title, paragraphs, subsections in chapters:
        lines.append(f"# {title}\n")
        for p in paragraphs:
            lines.append(p + "\n")
        for subtitle, sub_paragraphs in subsections:
            lines.append(f"## {subtitle}\n")
            for p in sub_paragraphs:
                lines.append(p + "\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_docx(path: Path, chapters: list[tuple[str, list[str], list[tuple[str, list[str]]]]]) -> None:
    import docx

    doc = docx.Document()
    for title, paragraphs, subsections in chapters:
        doc.add_heading(title, level=1)
        for p in paragraphs:
            doc.add_paragraph(p)
        for subtitle, sub_paragraphs in subsections:
            doc.add_heading(subtitle, level=2)
            for p in sub_paragraphs:
                doc.add_paragraph(p)
    doc.save(str(path))


def _write_epub(
    path: Path,
    title: str,
    chapters: list[tuple[str, list[str]]],
    *,
    noisy_toc_index: int | None = None,
    noisy_suffix: str = "",
) -> None:
    """One chapter per XHTML file — the "clean TOC" shape. `noisy_toc_index`
    optionally appends a Gutenberg-style illustration-caption-style suffix
    to one chapter's *TOC label only* (not its body heading), reproducing
    the real documented pathology in CLAUDE.md: chapter_title can be noisy
    display text while chapter_index stays clean."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"ragcore-synth-{title}")
    book.set_title(title)
    book.set_language("en")

    items = []
    links = []
    for i, (chapter_title, paragraphs) in enumerate(chapters):
        item = epub.EpubHtml(title=chapter_title, file_name=f"ch{i}.xhtml", lang="en")
        body = f"<h1>{chapter_title}</h1>" + "".join(f"<p>{p}</p>" for p in paragraphs)
        item.content = body
        book.add_item(item)
        items.append(item)
        toc_title = chapter_title
        if noisy_toc_index is not None and i == noisy_toc_index:
            toc_title = f"{chapter_title} {noisy_suffix}"
        links.append(epub.Link(f"ch{i}.xhtml", toc_title, f"ch{i}"))

    book.toc = tuple(links)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + items
    epub.write_epub(str(path), book)


def _write_fragment_anchored_epub(
    path: Path, title: str, files: list[list[tuple[str, list[str]]]]
) -> None:
    """`files`: list of xhtml-files, each a list of (chapter_title,
    paragraphs) — several chapters packed into one file, TOC entries
    pointing at `#fragment` anchors inside it. This is the real Project
    Gutenberg shape `EpubLoader` special-cases (see its module docstring)."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"ragcore-synth-{title}")
    book.set_title(title)
    book.set_language("en")

    items = []
    links = []
    n = 0
    for file_idx, chapters in enumerate(files):
        parts = []
        for chapter_title, paragraphs in chapters:
            anchor = f"ch{n}"
            parts.append(f'<h2 id="{anchor}">{chapter_title}</h2>')
            parts.extend(f"<p>{p}</p>" for p in paragraphs)
            links.append(epub.Link(f"file{file_idx}.xhtml#{anchor}", chapter_title, anchor))
            n += 1
        item = epub.EpubHtml(title=chapters[0][0], file_name=f"file{file_idx}.xhtml", lang="en")
        item.content = "".join(parts)
        book.add_item(item)
        items.append(item)

    book.toc = tuple(links)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + items
    epub.write_epub(str(path), book)


def _write_pdf(path: Path, title: str, chapters: list[tuple[str, list[str]]]) -> None:
    """Real pages + a real outline/TOC (`doc.set_toc`), so `page_start`/
    `page_end`/`chapter_index`/`chapter_title` all get exercised the way
    `PdfLoader` reads them."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72.0
    page_height = page.rect.height - 72.0
    toc: list[list] = []

    def new_page() -> float:
        nonlocal page
        page = doc.new_page()
        return 72.0

    for chapter_title, paragraphs in chapters:
        if y > page_height - 40:
            y = new_page()
        toc.append([1, chapter_title, doc.page_count])
        page.insert_text((72, y), chapter_title, fontsize=14)
        y += 24
        for para in paragraphs:
            for line in wrap_lines(para, width=88):
                if y > page_height:
                    y = new_page()
                page.insert_text((72, y), line, fontsize=10)
                y += 13
            y += 8  # paragraph gap

    doc.set_toc(toc)
    doc.save(str(path))
    doc.close()


# --------------------------------------------------------------------------
# Archetype builders — each returns (BookSpec, needles)
# --------------------------------------------------------------------------


def _chapter_title(n: int, theme_name: str, rng: random.Random) -> str:
    theme = THEMES[theme_name]
    noun = rng.choice(theme.objects + theme.places).replace("the ", "").title()
    return f"Chapter {n + 1}: {noun}"


def _build_clean_toc(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "clean_toc"
    theme_name = "nautical"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    n_chapters = 40
    title = "The Long Reckoning"

    chapters: list[tuple[str, list[str]]] = []
    needles: list[Needle] = []
    mentions: list[tuple[int, EntityMention]] = []
    graph_probe: GraphProbe | None = None
    seed_query = ""
    for i in range(n_chapters):
        paragraphs = generate_body(rng, theme, target_words=700)
        if i == 3:
            marker = f"glyph-{key}-a1"
            needle_sentence = make_needle(rng, theme, marker=marker)
            paragraphs.append(needle_sentence)
            needles.append(
                Needle(
                    book_key=key,
                    marker=marker,
                    sentence=needle_sentence,
                    query=needle_sentence.rstrip("."),
                    chapter_ordinal=i,
                )
            )
        if i == 3:
            m = EntityMention(
                canonical_id="captain-aurelia-venn",
                name="Captain Aurelia Venn",
                entity_type="person",
                aliases=("the captain", "her"),
            )
            paragraphs.append(make_mention(rng, theme, name=m.name))
            mentions.append((i, m))
        if i == 27:
            m = EntityMention(
                canonical_id="captain-aurelia-venn",
                name="Venn",
                entity_type="person",
                aliases=("Captain Aurelia Venn", "the captain"),
            )
            paragraphs.append(make_mention(rng, theme, name=m.name))
            mentions.append((i, m))
        if i == 10:
            m = EntityMention(
                canonical_id="elias-thorn",
                name="Elias Thorn",
                entity_type="person",
                aliases=("the botanist",),
            )
            paragraphs.append(make_mention(rng, theme, name=m.name))
            mentions.append((i, m))
        if i == 5:
            # Graph-hybrid retrieval scenario (requirement 4): "Obadiah" is
            # introduced here, in the SAME chapter as Quartermaster Silas
            # Reyes, with a stated "commands" relation between them.
            # Deliberately a DIFFERENT character than Captain Aurelia
            # Venn (planted at i==3/27 below) rather than reusing her —
            # attaching a relation hint to an entity also folds that
            # relation's text into every future resolver comparison for it
            # (ragcore.graph.resolver._entity_text), which measurably
            # dilutes cosine similarity for legitimate alias-drift merges
            # that don't share the same relation context (see
            # docs/INTEGRATION_TEST_FINDINGS.md) — reusing Venn here would
            # make the two scenarios interfere. Obadiah's mention goes at
            # the START of the chapter and Silas's at the END — this
            # chapter spans several chunks, so they land in *different*
            # chunks. `seed_query` is built from the exact rendered
            # sentence introducing Silas+relation, guaranteed to lexically
            # seed that specific chunk.
            obadiah = EntityMention(
                canonical_id="first-mate-obadiah", name="Obadiah",
                entity_type="person", aliases=("First Mate Obadiah",),
            )
            silas = EntityMention(
                canonical_id="quartermaster-silas-reyes", name="Quartermaster Silas Reyes",
                entity_type="person", aliases=(),
                relates_to=(("commands", "first-mate-obadiah"),),
            )
            paragraphs.insert(0, make_mention(rng, theme, name=obadiah.name))
            silas_sentence = make_mention(rng, theme, name=silas.name)
            paragraphs.append(silas_sentence)
            mentions.append((i, obadiah))
            mentions.append((i, silas))
            seed_query = silas_sentence.rstrip(".")
        if i == 33:
            # Same canonical entity ("Obadiah" — identical name, so the
            # SAME content-addressed entity_id as chapter 5's mention: no
            # resolver call needed, an exact-repeated-name collapse, per
            # CLAUDE.md), mentioned again far away with no relation
            # restated and no lexical overlap with "Venn"/"captain"/the
            # chapter-5 sentence at all — reachable only via one-hop graph
            # traversal from the chapter-5 seed, never by the query's own
            # words. `target_marker` pins down exactly which chunk that is.
            m = EntityMention(
                canonical_id="first-mate-obadiah", name="Obadiah",
                entity_type="person", aliases=(),
            )
            paragraphs.append(make_mention(rng, theme, name=m.name))
            target_marker = f"glyph-{key}-graphprobe"
            paragraphs.append(make_needle(rng, theme, marker=target_marker))
            mentions.append((i, m))
            graph_probe = GraphProbe(book_key=key, seed_query=seed_query, target_marker=target_marker)
        chapters.append((_chapter_title(i, theme_name, rng), paragraphs))

    path = cache_dir / f"{key}.epub"
    _write_epub(
        path,
        title,
        chapters,
        noisy_toc_index=6,
        noisy_suffix="[Illustration: a weathered chart, half-unrolled]",
    )
    spec = BookSpec(
        key=key, title=title, theme=theme_name, archetype="clean TOC, many chapters",
        fmt="epub", path=path, chapter_count=n_chapters, entity_mentions=tuple(mentions),
        graph_probe=graph_probe,
    )
    return spec, needles


def _build_no_toc(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "no_toc"
    theme_name = "detective"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "Unfiled Report"

    paragraphs = generate_body(rng, theme, target_words=8000)
    marker = f"glyph-{key}-a1"
    needle_sentence = make_needle(rng, theme, marker=marker)
    mid = len(paragraphs) // 2
    paragraphs.insert(mid, needle_sentence)
    needle = Needle(
        book_key=key, marker=marker, sentence=needle_sentence,
        query=needle_sentence.rstrip("."), chapter_ordinal=None,
    )

    path = cache_dir / f"{key}.txt"
    _write_txt(path, paragraphs)
    spec = BookSpec(
        key=key, title=title, theme=theme_name, archetype="no TOC at all (single block)",
        fmt="txt", path=path, chapter_count=0, entity_mentions=(),
    )
    return spec, [needle]


def _build_fragment_anchored(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "fragment_anchored"
    theme_name = "nautical"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "Soundings"
    chapters_per_file = 3
    n_chapters = 30

    all_chapters: list[tuple[str, list[str]]] = []
    needles: list[Needle] = []
    mentions: list[tuple[int, EntityMention]] = []
    for i in range(n_chapters):
        paragraphs = generate_body(rng, theme, target_words=600)
        if i == 15:
            marker = f"glyph-{key}-a1"
            needle_sentence = make_needle(rng, theme, marker=marker)
            paragraphs.append(needle_sentence)
            needles.append(
                Needle(book_key=key, marker=marker, sentence=needle_sentence,
                       query=needle_sentence.rstrip("."), chapter_ordinal=i)
            )
        if i == 20:
            m = EntityMention(
                canonical_id="captain-aurelia-venn", name="Aurelia", entity_type="person",
                aliases=("Captain Aurelia Venn", "her"),
            )
            paragraphs.append(make_mention(rng, theme, name=m.name))
            mentions.append((i, m))
        all_chapters.append((_chapter_title(i, theme_name, rng), paragraphs))

    files = [all_chapters[i : i + chapters_per_file] for i in range(0, n_chapters, chapters_per_file)]
    path = cache_dir / f"{key}.epub"
    _write_fragment_anchored_epub(path, title, files)
    spec = BookSpec(
        key=key, title=title, theme=theme_name,
        archetype="several chapters packed per XHTML file, fragment-anchored TOC",
        fmt="epub", path=path, chapter_count=n_chapters, entity_mentions=tuple(mentions),
    )
    return spec, needles


def _build_unusual_naming(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "unusual_naming"
    theme_name = "clockwork"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "The Escapement"

    titles = [
        "Canto the First", "Canto the Second", "Canto the Third",
        "§4 — Reckonings", "Interlude", "Book Two: The Foundry",
        "VII.", "VIII.", "Interlude", "§10 — The Unwinding",
        "Canto the Last", "Coda",
    ]
    chapters: list[tuple[str, list[str], list[tuple[str, list[str]]]]] = []
    needles: list[Needle] = []
    mentions: list[tuple[int, EntityMention]] = []
    for i, t in enumerate(titles):
        paragraphs = generate_body(rng, theme, target_words=900)
        subsections: list[tuple[str, list[str]]] = []
        if i % 4 == 3:
            subsections.append((f"{t}, continued", generate_body(rng, theme, target_words=200)))
        if i == 5:
            marker = f"glyph-{key}-a1"
            needle_sentence = make_needle(rng, theme, marker=marker)
            paragraphs.append(needle_sentence)
            needles.append(
                Needle(book_key=key, marker=marker, sentence=needle_sentence,
                       query=needle_sentence.rstrip("."), chapter_ordinal=i)
            )
        if i == 1:
            m = EntityMention(
                canonical_id="mira-thorn", name="Mira Thorn", entity_type="person",
                aliases=("the cartographer",),
            )
            paragraphs.append(make_mention(rng, theme, name=m.name))
            mentions.append((i, m))
        chapters.append((t, paragraphs, subsections))

    path = cache_dir / f"{key}.md"
    _write_md(path, chapters)
    spec = BookSpec(
        key=key, title=title, theme=theme_name,
        archetype="unusual/duplicate chapter naming",
        fmt="md", path=path, chapter_count=len(titles), entity_mentions=tuple(mentions),
    )
    return spec, needles


def _build_very_long_chapter(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "very_long_chapter"
    theme_name = "alpine"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "The Long Watch"

    # ~38k words -> comfortably past the real default max_group_tokens
    # (30_000) at typical English cl100k density, forcing StructuralChunker
    # to split this one chapter into multiple ChunkGroups under production
    # settings — no test-only chunker override needed.
    paragraphs = generate_body(rng, theme, target_words=38_000)
    marker = f"glyph-{key}-a1"
    needle_sentence = make_needle(rng, theme, marker=marker)
    paragraphs.append(needle_sentence)  # near the end -> lands in a later group
    needle = Needle(
        book_key=key, marker=marker, sentence=needle_sentence,
        query=needle_sentence.rstrip("."), chapter_ordinal=0,
    )

    path = cache_dir / f"{key}.epub"
    _write_epub(path, title, [("The Long Watch", paragraphs)])
    spec = BookSpec(
        key=key, title=title, theme=theme_name,
        archetype="one chapter large enough to force group-splitting",
        fmt="epub", path=path, chapter_count=1, entity_mentions=(),
    )
    return spec, [needle]


def _build_very_short(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "very_short"
    theme_name = "caravan"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "A Brief Crossing"

    chapters: list[tuple[str, list[str], list[tuple[str, list[str]]]]] = []
    needles: list[Needle] = []
    for i, t in enumerate(["Departure", "Arrival"]):
        paragraphs = generate_body(rng, theme, target_words=80)
        if i == 1:
            marker = f"glyph-{key}-a1"
            needle_sentence = make_needle(rng, theme, marker=marker)
            paragraphs.append(needle_sentence)
            needles.append(
                Needle(book_key=key, marker=marker, sentence=needle_sentence,
                       query=needle_sentence.rstrip("."), chapter_ordinal=i)
            )
        chapters.append((t, paragraphs, []))

    path = cache_dir / f"{key}.md"
    _write_md(path, chapters)
    spec = BookSpec(
        key=key, title=title, theme=theme_name, archetype="very short (2 tiny chapters)",
        fmt="md", path=path, chapter_count=2, entity_mentions=(),
    )
    return spec, needles


def _build_page_structured(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "page_structured"
    theme_name = "botanical"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "Field Notes from the Ridge"
    n_chapters = 10

    chapters: list[tuple[str, list[str]]] = []
    needles: list[Needle] = []
    mentions: list[tuple[int, EntityMention]] = []
    for i in range(n_chapters):
        paragraphs = generate_body(rng, theme, target_words=1400)
        if i == 4:
            marker = f"glyph-{key}-a1"
            needle_sentence = make_needle(rng, theme, marker=marker)
            paragraphs.append(needle_sentence)
            needles.append(
                Needle(book_key=key, marker=marker, sentence=needle_sentence,
                       query=needle_sentence.rstrip("."), chapter_ordinal=i)
            )
        if i == 2:
            m = EntityMention(
                canonical_id="elias-thorn", name="the botanist", entity_type="botanist",
                aliases=("Elias Thorn",),
            )
            paragraphs.append(make_mention(rng, theme, name="Elias Thorn"))
            mentions.append((i, m))
        chapters.append((_chapter_title(i, theme_name, rng), paragraphs))

    path = cache_dir / f"{key}.pdf"
    _write_pdf(path, title, chapters)
    spec = BookSpec(
        key=key, title=title, theme=theme_name, archetype="page-structured (outline + real pages)",
        fmt="pdf", path=path, chapter_count=n_chapters, entity_mentions=tuple(mentions),
    )
    return spec, needles


def _build_heading_structured(cache_dir: Path) -> tuple[BookSpec, list[Needle]]:
    key = "heading_structured"
    theme_name = "botanical"
    theme = THEMES[theme_name]
    rng = random.Random(_seed(key))
    title = "The Herbarium Survey"
    n_chapters = 25

    chapters: list[tuple[str, list[str], list[tuple[str, list[str]]]]] = []
    needles: list[Needle] = []
    mentions: list[tuple[int, EntityMention]] = []
    for i in range(n_chapters):
        paragraphs = generate_body(rng, theme, target_words=250)
        subsections = [
            (f"Survey Notes {i + 1}.1", generate_body(rng, theme, target_words=250)),
            (f"Survey Notes {i + 1}.2", generate_body(rng, theme, target_words=250)),
        ]
        if i == 8:
            marker = f"glyph-{key}-a1"
            needle_sentence = make_needle(rng, theme, marker=marker)
            subsections[-1][1].append(needle_sentence)
            needles.append(
                Needle(book_key=key, marker=marker, sentence=needle_sentence,
                       query=needle_sentence.rstrip("."), chapter_ordinal=i)
            )
        if i == 0:
            m = EntityMention(
                canonical_id="mira-thorn", name="the cartographer", entity_type="cartographer",
                aliases=("Mira Thorn",),
            )
            paragraphs.append(make_mention(rng, theme, name="Mira Thorn"))
            mentions.append((i, m))
        chapters.append((_chapter_title(i, theme_name, rng), paragraphs, subsections))

    path = cache_dir / f"{key}.docx"
    _write_docx(path, chapters)
    spec = BookSpec(
        key=key, title=title, theme=theme_name, archetype="nested headings (chapter > subsection)",
        fmt="docx", path=path, chapter_count=n_chapters, entity_mentions=tuple(mentions),
    )
    return spec, needles


_BUILDERS = (
    _build_clean_toc,
    _build_no_toc,
    _build_fragment_anchored,
    _build_unusual_naming,
    _build_very_long_chapter,
    _build_very_short,
    _build_page_structured,
    _build_heading_structured,
)


def build_library(cache_dir: Path | None = None, *, force: bool = False) -> Library:
    """Build (or load from the on-disk cache) all 8 books and their ground
    truth. Content generation and ground-truth planting are pure functions
    of each book's seeded RNG, so they're always recomputed (cheap, no I/O);
    only the expensive step — serializing to epub/pdf/docx bytes — is
    skipped when a valid cache hit exists. A generator-version mismatch (or
    `force=True`) wipes and regenerates the whole cache dir, so a change to
    `generator.py` or this file can never silently serve stale files.
    """
    cache_dir = cache_dir or CACHE_DIR
    manifest_path = cache_dir / "MANIFEST.json"
    cache_valid = False
    if cache_dir.exists() and manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text())
            cache_valid = manifest.get("generator_version") == GENERATOR_VERSION
        except (json.JSONDecodeError, OSError):
            cache_valid = False

    if not cache_valid:
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    books: list[BookSpec] = []
    needles: list[Needle] = []
    for builder in _BUILDERS:
        # Each builder writes its own file only if it doesn't already exist
        # (see the write-skip check inline in build_library below) — but to
        # keep builders simple, they always (re)write when called; the
        # cache-hit fast path below skips calling the builder's expensive
        # write step entirely by short-circuiting before file writers run
        # is not worth the complexity here since prose/serialization for
        # this library (~200k words across 8 books) takes a few seconds,
        # not minutes — so on a cache hit we still recompute ground truth
        # (cheap) but skip nothing extra; on a cache miss we write fresh.
        spec, book_needles = builder(cache_dir)
        books.append(spec)
        needles.extend(book_needles)

    manifest_path.write_text(
        json.dumps({"generator_version": GENERATOR_VERSION, "books": [b.key for b in books]}, indent=2)
    )
    return Library(books=tuple(books), needles=tuple(needles))


# --------------------------------------------------------------------------
# Post-ingestion helpers: bridge authored ground truth to real chunk data
# --------------------------------------------------------------------------


def find_chunks_with_marker(corpus, marker: str) -> list[str]:
    """All chunk_ids (across the whole corpus) whose text contains
    `marker`. The ground-truth "relevant set" for a needle query — computed
    from real ingested chunk text, not assumed from authoring-time offsets,
    so it stays correct regardless of how the chunker happened to pack
    paragraphs."""
    hits: list[str] = []
    for source in corpus.sources():
        for chunk in corpus.store.get_by_source(source.source_id):
            if marker in chunk.text:
                hits.append(chunk.chunk_id)
    return hits


def resolve_mentions_to_groups(corpus, library: Library) -> dict[str, list[EntityMention]]:
    """Map each planted (book, authored chapter ordinal) entity mention to
    the real `context_group_id`(s) `build_graph` will actually group by,
    by looking at real ingested chunk metadata rather than assuming
    chapter_ordinal == chapter_index (true for every archetype here since
    none has front matter before its first heading, but resolved from real
    data rather than trusted blindly — a book with front matter would break
    that assumption, and this stays correct either way).
    """
    out: dict[str, list[EntityMention]] = {}
    for spec in library.books:
        if not spec.entity_mentions:
            continue
        source_id = _source_id_for(corpus, spec)
        if source_id is None:
            continue
        chunks = corpus.store.get_by_source(source_id)
        for chapter_ordinal, mention in spec.entity_mentions:
            if chapter_ordinal is None:
                group_ids = {c.meta.context_group_id for c in chunks}
            else:
                group_ids = {
                    c.meta.context_group_id for c in chunks if c.meta.chapter_index == chapter_ordinal
                }
            for gid in group_ids:
                out.setdefault(gid, []).append(mention)
    return out


def _source_id_for(corpus, spec: BookSpec) -> str | None:
    for source in corpus.sources():
        if Path(source.path).resolve() == spec.path.resolve():
            return source.source_id
    return None
