"""The public learning bundle contains reviewed content, not the private docs tree."""

import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "apps/learning-content/site"


class Links(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids, self.hrefs = set(), []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, attrs["id"]
            self.ids.add(attrs["id"])
        if "href" in attrs:
            self.hrefs.append(attrs["href"])


def builder():
    pytest.importorskip("markdown_it")
    pytest.importorskip("jinja2")
    spec = importlib.util.spec_from_file_location("learning_builder", ROOT / "scripts/build_learning_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_navigation_and_anchors_are_resolvable():
    concepts = Links((SITE / "concepts/index.html").read_text())
    catalog = json.loads((SITE / "diagrams/catalog.json").read_text())
    diagrams = {d["id"] for d in catalog}
    for page in [SITE / "concepts/index.html", SITE / "diagrams/index.html",
                 ROOT / "apps/learning-replay/index.html", ROOT / "apps/api-gateway/static/index.html"]:
        document = Links(page.read_text())
        for path in ("/", "/learn/", "/learn/concepts/", "/learn/diagrams/"):
            assert path in document.hrefs
        for href in document.hrefs:
            parsed = urlsplit(href)
            if parsed.path == "/learn/concepts/" and parsed.fragment:
                assert parsed.fragment in concepts.ids, href
            if parsed.path == "/learn/diagrams/" and parsed.fragment:
                assert parsed.fragment in diagrams, href
    for item in catalog:
        for href in item["related"]:
            assert urlsplit(href).fragment in concepts.ids


def test_diagrams_are_self_contained_and_match_the_catalog():
    catalog = json.loads((SITE / "diagrams/catalog.json").read_text())
    assert len(catalog) == len(json.loads((ROOT / "apps/learning-content/content.json").read_text())["diagrams"])
    for diagram in catalog:
        path = SITE / "diagrams/svg" / (diagram["id"] + ".svg")
        document = ET.fromstring(path.read_text())
        viewbox = list(map(float, document.attrib["viewBox"].split()))
        assert viewbox[2:] == [diagram["width"], diagram["height"]]
        assert diagram["width"] > 100 and diagram["height"] > 100
        assert "data:font" in path.read_text()
        for node in document.iter():
            assert node.tag.rsplit("}", 1)[-1] not in {"script", "foreignObject"}
            for key, value in node.attrib.items():
                assert not key.lower().startswith("on")
                if key.rsplit("}", 1)[-1] == "href":
                    assert value.startswith(("/learn/", "data:", "#", "https://", "http://"))
        source = ROOT / "docs/excalidraw" / (diagram["id"] + ".excalidraw")
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == diagram["source_sha256"]


def test_bundle_has_no_workstation_links_or_private_files():
    for path in SITE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix in {".html", ".json", ".svg", ".excalidraw"}
        content = path.read_text()
        for forbidden in ("vscode://", "obsidian://", "/home/", "/opt/projects/", "file://"):
            assert forbidden not in content, (path, forbidden)


def test_source_headings_are_preserved_when_source_available():
    source = ROOT / "docs/concepts_explanations.md"
    if not source.exists():
        pytest.skip("Private authoring source is not in this checkout")
    module = builder()
    _, _, headings, _ = module.parse_concepts(source.read_text())
    summary = json.loads((SITE / "concepts/build.json").read_text())
    assert summary["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    ids = Links((SITE / "concepts/index.html").read_text()).ids
    assert all(heading["id"] in ids for heading in headings)
    assert len(headings) == summary["headings"]


def test_markdown_disables_raw_html_and_preserves_code():
    module = builder()
    source = '## Gradient ^gradient\n\n<script>alert(1)</script>\n\n[[#^gradient|Gradient]]\n\n[code](vscode://file/private.py)\n\n```python\n[[not_a_link]]\n```\n'
    md, tokens, _, aliases = module.parse_concepts(source)
    module.rewrite_tokens(tokens, aliases, set())
    html = md.renderer.render(tokens, md.options, {})
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'href="/learn/concepts/#gradient"' in html
    assert '<span class="source-reference">code</span>' in html
    assert "[[not_a_link]]" in html
    for target in ("javascript:alert(1)", "file:///private", "../private.md", "data:text/html,boom"):
        assert module.resolve_link(target, aliases, set()) is None
    with pytest.raises(ValueError, match="Unknown concept"):
        module.resolve_link("#missing", aliases, set())


def test_scene_export_strips_local_state_and_rewrites_links():
    module = builder()
    original = {"elements": [{"type": "text", "text": "Example", "link": "concepts_explanations.md#^gradient"},
                             {"type": "text", "text": "Deleted", "isDeleted": True}],
                "appState": {"private": "do not publish"}, "files": {}}
    sanitized = module.sanitize_scene(original, {"gradient": "gradient"}, set())
    assert len(sanitized["elements"]) == 1
    assert sanitized["elements"][0]["link"] == "/learn/concepts/#gradient"
    assert "private" not in sanitized["appState"]
    assert original["elements"][0]["link"].endswith("#^gradient")
    with pytest.raises(ValueError, match="review"):
        module.sanitize_scene(dict(original, files={"image": {}}), {"gradient": "gradient"}, set())
