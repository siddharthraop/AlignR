"""load_facts: the launch values, each with a source ID and a unit, and the conflicts among them."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .corpus import ROOT, Corpus

PLACEHOLDER = re.compile(r"\{([a-z0-9][a-z0-9-]*)\}")

# Conversions to a base unit, so the same quantity stated in two units can be compared.
UNIT_BASE = {
    "GBps": ("Bps", 1e9), "TB/s": ("Bps", 1e12), "GiBps": ("Bps", 2**30), "TBps": ("Bps", 1e12),
    "GB": ("B", 1e9), "GiB": ("B", 2**30), "TB": ("B", 1e12),
    "Gbps": ("bps", 1e9), "Tbps": ("bps", 1e12),
    "TFLOPs": ("FLOPs", 1e12), "PFLOPs": ("FLOPs", 1e15), "TOPs": ("OPs", 1e12),
}
UNITS = sorted(set(UNIT_BASE) | {"chips", ""}, key=lambda u: (u == "", u))


@dataclass
class Value:
    quantity: str
    value: str
    unit: str
    source: str
    key: str = ""
    table: str = ""
    approx: bool = False


@dataclass
class Conflict:
    id: str
    kind: str            # "sources" (two facts disagree) or "siblings" (sibling topics disagree)
    quantity: str
    detail: str
    ask: str             # "engineer" or "writer"
    key: str = ""
    known_issue: str = ""


@dataclass
class Facts:
    generation: str
    product_key: str
    prefix: str
    platform: str
    siblings: list[str]
    page: str
    sources: dict[str, dict]
    values: list[Value]
    statements: list[dict]
    tables: dict[str, dict]
    new_topics: list[dict]
    conflicts: list[Conflict] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)   # values with no key yet: keydef proposals

    @property
    def keyed(self) -> dict[str, Value]:
        return {v.key: v for v in self.values if v.key}

    def allowed_keys(self) -> list[str]:
        return [self.product_key, "product-cloud-tpu"] + list(self.keyed)

    def spec_values(self) -> list[Value]:
        return [v for v in self.values if v.table == "spec"]

    def statement(self, fid: str) -> dict:
        return next(s for s in self.statements if s["id"] == fid)


def load_facts(path: str | Path, corpus: Corpus | None = None) -> Facts:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    d = yaml.safe_load(path.read_text())
    f = Facts(
        generation=d["generation"], product_key=d["product_key"], prefix=d["topic_prefix"],
        platform=d.get("platform", d["generation"]), siblings=list(d["siblings"]), page=d.get("page", ""),
        sources=d["sources"], values=[Value(**{k: str(v) if k in ("value",) else v for k, v in x.items()})
                                      for x in d["values"]],
        statements=d.get("statements", []), tables=d.get("tables", {}), new_topics=d.get("new_topics", []),
    )
    for v in f.values:
        if v.source not in f.sources:
            raise ValueError(f"value {v.quantity} cites unknown source {v.source}")
    if corpus is not None:
        for v in f.values:
            if v.key and v.key not in corpus.keys:
                f.proposals.append({"key": v.key, "value": v.value, "unit": v.unit, "source": v.source})
            elif v.key and corpus.keys[v.key] != v.value:
                raise ValueError(f"key {v.key} is {corpus.keys[v.key]!r} in the map but {v.value!r} in the facts")
    f.conflicts = source_conflicts(f)
    return f


def to_base(value: str, unit: str) -> tuple[str, float] | None:
    try:
        num = float(value.replace(",", ""))
    except ValueError:
        return None
    if unit not in UNIT_BASE:
        return None
    base, mult = UNIT_BASE[unit]
    return base, num * mult


def source_conflicts(f: Facts) -> list[Conflict]:
    """Two facts that state one quantity with different values once units are converted."""
    out = []
    by_q: dict[str, list[Value]] = {}
    for v in f.values:
        by_q.setdefault(v.quantity, []).append(v)
    for q, vs in by_q.items():
        if len(vs) < 2:
            continue
        a = vs[0]
        for b in vs[1:]:
            ba, bb = to_base(a.value, a.unit), to_base(b.value, b.unit)
            same = ba and bb and ba[0] == bb[0] and abs(ba[1] - bb[1]) / max(ba[1], bb[1]) < 1e-9
            if not same:
                approx = " (approximately)" if b.approx else ""
                out.append(Conflict(
                    id=f"C{len(out) + 1}", kind="sources", quantity=q, key=a.key or b.key, ask="engineer",
                    known_issue="KI-2" if q == "hbm-bandwidth" else "",
                    detail=(f"{a.source} gives {q} as {a.value} {a.unit}; {b.source} gives {b.value} {b.unit}{approx}. "
                            f"{a.value} {a.unit} is {_as(a, b.unit)} {b.unit}, so the two sources disagree.")))
    return out


def _as(a: Value, unit: str) -> str:
    ba = to_base(a.value, a.unit)
    if not ba or unit not in UNIT_BASE:
        return "?"
    return f"{ba[1] / UNIT_BASE[unit][1]:.2f}".rstrip("0").rstrip(".")


def fill(text: str, keys: dict[str, str]) -> str:
    """Replace {key} placeholders with the key values, as a reader would see them."""
    return PLACEHOLDER.sub(lambda m: keys.get(m.group(1), m.group(0)), text)
