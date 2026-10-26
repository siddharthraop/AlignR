"""Compare a draft with the topic Google published: the edit ratio and a reader's view of both.

This is the evaluator. It runs after drafting and is the only code that reads the held-out topics.

Edit ratio: both topics are rendered as a reader sees them (keys resolved, reused notes expanded, draft comments
left out) and split into word and punctuation tokens. The ratio is the number of tokens a writer must insert,
delete or replace to turn the draft into the published text, divided by the length of the longer of the two.
0% means identical; 100% means nothing in common.
"""
from __future__ import annotations

import difflib
import re

from lxml import etree

from .corpus import Corpus, plain
from .skeleton import blocks, inline

TOKENS = re.compile(r"\w+(?:[./-]\w+)*|[^\w\s]")


def tokens(text: str) -> list[str]:
    return TOKENS.findall(text.lower())


def edit_ratio(draft: str, published: str) -> float:
    a, b = tokens(draft), tokens(published)
    if not a and not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    edits = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return round(edits / max(len(a), len(b)), 4)


def reader_view(root: etree._Element, corpus: Corpus) -> list[dict]:
    """The topic as blocks a page can show: title, shortdesc, sections, paragraphs, notes, tables, questions."""
    k, n = corpus.keys, corpus.notes
    out = [{"kind": "title", "text": plain(root.find("title"), k)},
           {"kind": "shortdesc", "text": plain(root.find("shortdesc"), k)}]
    for sid, sec, kids in blocks(root):
        if sec is not None:
            out.append({"kind": "section", "text": plain(sec.find("title"), k)})
        for c in kids:
            if c.tag == "p":
                out.append({"kind": "p", "text": plain(c, k)})
            elif c.tag == "note":
                out.append({"kind": "note", "text": plain(c, k, n)})
            elif c.tag == "table":
                head = [plain(e, k) for e in c.find(".//thead/row").findall("entry")]
                rows = [[plain(e, k) for e in r.findall("entry")] for r in c.find(".//tbody").findall("row")]
                out.append({"kind": "table", "head": head, "rows": rows})
            elif c.tag == "draft-comment":
                out.append({"kind": "question", "text": " ".join("".join(c.itertext()).split())})
            elif c.tag == "section":
                pass
    # Draft comments inside sections
    for dc in root.iter("draft-comment"):
        t = " ".join("".join(dc.itertext()).split())
        if not any(b.get("kind") == "question" and b.get("text") == t for b in out):
            out.append({"kind": "question", "text": t})
    return out


def compare(draft_xml: bytes, published_href: str | None, corpus: Corpus) -> dict:
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False, remove_comments=True)
    d = etree.fromstring(draft_xml, parser)
    dtext = plain(d, corpus.keys, corpus.notes)
    out = {"draft_view": reader_view(d, corpus), "draft_text": dtext}
    if not published_href:
        return out | {"published": None, "edit_ratio": None}
    try:
        p = corpus.published(published_href)
    except OSError:
        return out | {"published": None, "edit_ratio": None}
    ptext = plain(p, corpus.keys, corpus.notes)
    return out | {"published": published_href, "published_view": reader_view(p, corpus), "published_text": ptext,
                  "edit_ratio": edit_ratio(dtext, ptext), "published_title": inline(p.find("title"))}
