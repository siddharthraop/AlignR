"""The help gate: every help ID on the page has a help topic, and every help topic is in the map."""
import re
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
P = etree.XMLParser(load_dtd=False, no_network=True)


def test_every_help_id_on_the_page_has_a_topic():
    page = (ROOT / "app" / "index.html").read_text() + (ROOT / "app" / "app.js").read_text()
    used = set(re.findall(r'data-help="([a-z-]+)"', page)) | set(re.findall(r'helpId = "([a-z-]+)"', page))
    ids = {r.get("appid") for f in (ROOT / "help" / "topics").glob("*.dita")
           for r in etree.parse(str(f), P).iter("resourceid")}
    assert used and used <= ids, used - ids


def test_every_topic_is_in_the_help_map_and_the_guide():
    topics = {f"topics/{f.name}" for f in (ROOT / "help" / "topics").glob("*.dita")}
    for m in ("alignr-help.ditamap", "alignr-user-guide.bookmap"):
        refs = {e.get("href") for e in etree.parse(str(ROOT / "help" / m), P).iter() if e.get("href", "").startswith("topics/")}
        assert topics <= refs, (m, topics - refs)
