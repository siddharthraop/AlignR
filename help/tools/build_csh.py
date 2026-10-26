"""Build csh.json: each console help ID -> its help topic's URL, title, short description, first steps
and related topics. The console's help panel reads it.

    python help/tools/build_csh.py OUT_DIR

Help IDs come from <resourceid appname="manifest-console" appid="..."/> in each topic's prolog, so
the mapping lives with the topic a writer maintains, not in the UI code. Related topics come from the
map's relationship table, as DITA-OT renders them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lxml import etree

HELP = Path(__file__).resolve().parent.parent
MAP = HELP / "alignr-help.ditamap"
APP = "alignr-app"
PARSER = etree.XMLParser(load_dtd=False, resolve_entities=False, no_network=True)


def keys() -> dict[str, str]:
    tree = etree.parse(str(HELP / "keys" / "alignr-keys.ditamap"), PARSER)
    out = {}
    for kd in tree.iter("keydef"):
        word = kd.find(".//keyword")
        link = kd.find(".//linktext")
        out[kd.get("keys", "")] = (
            word.text if word is not None else link.text if link is not None else ""
        ) or ""
    return out


def text(el: etree._Element | None, k: dict[str, str], titles: dict[str, str]) -> str:
    """Plain text with key and cross-reference text resolved, as a reader sees it."""
    if el is None:
        return ""
    parts: list[str] = [el.text or ""]
    for child in el:
        if child.get("keyref") and not len(child) and not (child.text or "").strip():
            parts.append(k.get(child.get("keyref", ""), ""))
        elif child.tag == "xref" and not (child.text or "").strip() and child.get("href"):
            parts.append(titles.get(Path(child.get("href", "")).name, child.get("href", "")))
        else:
            parts.append(text(child, k, titles))
        parts.append(child.tail or "")
    return " ".join("".join(parts).split())


def topics() -> list[Path]:
    tree = etree.parse(str(MAP), PARSER)
    hrefs = [t.get("href", "") for t in tree.iter("topicref") if t.get("href", "").startswith("topics/")]
    return list(dict.fromkeys(HELP / h for h in hrefs))


def related(name: str) -> list[str]:
    tree = etree.parse(str(MAP), PARSER)
    out: list[str] = []
    for row in tree.iter("relrow"):
        cells = [
            [Path(t.get("href", "")).name for t in cell.iter("topicref")] for cell in row.iter("relcell")
        ]
        if any(name in c for c in cells):
            out += [h for c in cells for h in c if h != name]
    return list(dict.fromkeys(out))


def build() -> dict[str, Any]:
    k = keys()
    parsed = {p.name: etree.parse(str(p), PARSER).getroot() for p in topics()}
    titles = {n: text(r.find("title"), k, {}) for n, r in parsed.items()}
    csh: dict[str, Any] = {}
    for name, root in parsed.items():
        for rid in root.iter("resourceid"):
            if rid.get("appname") != APP:
                continue
            appid = rid.get("appid", "")
            if appid in csh:
                raise SystemExit(f"help ID {appid!r} is on two topics: {csh[appid]['href']} and {name}")
            csh[appid] = {
                "href": f"topics/{Path(name).with_suffix('.html')}",
                "title": titles[name],
                "shortdesc": text(root.find("shortdesc"), k, titles),
                "steps": [text(c, k, titles) for c in root.iter("cmd")][:4],
                "related": [{"title": titles[r], "href": f"topics/{Path(r).with_suffix('.html')}"}
                            for r in related(name) if r in titles],
            }  # fmt: skip
    return dict(sorted(csh.items()))


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HELP / "out"
    out.mkdir(parents=True, exist_ok=True)
    data = build()
    (out / "csh.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'csh.json'}: {len(data)} help IDs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
