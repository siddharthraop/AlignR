"""Read the DITA set: the map, the keys, the reuse warehouse and sibling topics.

The held-out generation is fenced off here. `Corpus(holdout="tpu7x-")` refuses to open any topic whose file
name starts with the holdout prefix, and never opens the comparison topic that lists every generation. Only
`alignr.compare`, which runs after drafting, reads the published topics, through `published()`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

PARSER = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False, remove_comments=True)
ROOT = Path(__file__).resolve().parent.parent
# Topics that state the held-out generation's values or prose, so they never reach the model.
ALWAYS_EXCLUDED = {"tpu-generations-compared.dita"}


def corpus_dir() -> Path:
    return Path(os.environ.get("ALIGNR_CORPUS", ROOT / "corpus"))


class HoldoutError(PermissionError):
    """Raised when drafting code tries to read the published topic it is meant to draft."""


@dataclass
class Family:
    generation: str          # v6e
    prefix: str              # tpu-v6e-
    slots: dict[str, str]    # slot -> href relative to the map, e.g. "specifications" -> topics/tpu-v6e-specifications.dita


@dataclass
class Note:
    id: str
    type: str
    xml: str                 # the note as written in the warehouse
    text: str                # plain text, keys resolved


@dataclass
class Corpus:
    holdout: str = ""        # file-name prefix of the held-out generation, e.g. "tpu7x-"
    root: Path = field(default_factory=lambda: corpus_dir() / "tpu")

    def __post_init__(self):
        self.map_path = self.root / "tpu.ditamap"
        self.keys = self._keys()
        self.families = self._families()
        self.notes = self._notes()

    # ---- reading -------------------------------------------------------------------------------------------
    def _guard(self, path: Path):
        name = path.name
        if name in ALWAYS_EXCLUDED or (self.holdout and name.startswith(self.holdout)):
            raise HoldoutError(f"{name} is held out from drafting")

    def parse(self, href: str) -> etree._Element:
        path = (self.root / href).resolve()
        self._guard(path)
        return etree.parse(str(path), PARSER).getroot()

    def published(self, href: str) -> etree._Element:
        """Evaluator only: read a held-out topic to compare a draft with it."""
        return etree.parse(str((self.root / href).resolve()), PARSER).getroot()

    def _keys(self) -> dict[str, str]:
        tree = etree.parse(str(self.root / "keys" / "tpu-keys.ditamap"), PARSER)
        out = {}
        for kd in tree.iter("keydef"):
            word = kd.find(".//keyword")
            out[kd.get("keys")] = (word.text or "") if word is not None else ""
        return out

    def key_line(self, key: str) -> int:
        for i, line in enumerate((self.root / "keys" / "tpu-keys.ditamap").read_text().splitlines(), 1):
            if f'keys="{key}"' in line:
                return i
        return 0

    def _families(self) -> dict[str, Family]:
        tree = etree.parse(str(self.map_path), PARSER)
        fams = {}
        for tr in tree.iter("topicref"):
            if tr.get("collection-type") != "family" or not tr.get("platform"):
                continue
            gen = tr.get("platform")
            head = tr.get("href")
            stem = Path(head).stem
            prefix = stem[: stem.rindex("-") + 1]  # tpu-v6e-intro -> tpu-v6e-
            slots = {stem[len(prefix):]: head}
            for child in tr.findall("topicref"):
                cs = Path(child.get("href")).stem
                if cs.startswith(prefix):
                    slots[cs[len(prefix):]] = child.get("href")
            fams[gen] = Family(gen, prefix, slots)
        return fams

    def _notes(self) -> dict[str, Note]:
        root = etree.parse(str(self.root / "reuse" / "tpu-warehouse.dita"), PARSER).getroot()
        out = {}
        for n in root.iter("note"):
            out[n.get("id")] = Note(n.get("id"), n.get("type", "note"), etree.tostring(n, encoding="unicode").strip(),
                                    plain(n, self.keys))
        return out

    # ---- tools ---------------------------------------------------------------------------------------------
    def find_siblings(self, slot: str, generations: list[str]) -> list[tuple[str, str]]:
        """Topics of the same type in the same place in the map for earlier generations: [(generation, href)]."""
        out = []
        for g in generations:
            fam = self.families.get(g)
            if fam and slot in fam.slots:
                out.append((g, fam.slots[slot]))
        return out


def plain(el: etree._Element | None, keys: dict[str, str], notes: dict[str, "Note"] | None = None,
          skip=("draft-comment", "prolog")) -> str:
    """Text as a reader sees it: keys resolved, conref notes expanded, draft comments left out."""
    if el is None:
        return ""
    out: list[str] = []

    def walk(e):
        if not isinstance(e.tag, str) or e.tag in skip:
            return
        if e.get("conref") and notes is not None:
            frag = e.get("conref").split("/")[-1]
            if frag in notes:
                out.append(" " + notes[frag].text + " ")
            return
        if e.get("keyref") and not len(e) and not (e.text or "").strip():
            out.append(keys.get(e.get("keyref"), ""))
        else:
            out.append(e.text or "")
            for c in e:
                walk(c)
                out.append(c.tail or "")
        if e.tag in ("entry", "p", "title", "shortdesc", "note", "row", "li"):
            out.append(" ")

    walk(el)
    return re.sub(r"\s+", " ", "".join(out)).strip()
