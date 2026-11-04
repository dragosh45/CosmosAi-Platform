"""The public learning bundle contains only reviewed, allowlisted source content."""

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
        for path in ("/", "/learn/", "/learn/concepts/", "/learn/courses/", "/learn/diagrams/"):
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
        assert path.suffix in {".html", ".json", ".svg", ".excalidraw", ".png", ".js"}
        if path.suffix == ".png":
            summary = json.loads((SITE / "concepts/build.json").read_text())
            records = {item["url"]: item for item in summary.get("images", [])}
            record = records["/learn/" + str(path.relative_to(SITE))]
            assert record["license"] in {"own", "cc-by", "public-domain"}
            manifest = json.loads((ROOT / "apps/learning-content/content.json").read_text())
            assert any(item["src"] == record["src"] and item["license"] == record["license"]
                       for item in manifest.get("images", []))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            continue
        content = path.read_text()
        for forbidden in ("vscode://", "obsidian://", "/home/", "/Users/", "/opt/projects/", "file://"):
            assert forbidden not in content, (path, forbidden)


def test_source_headings_are_preserved_when_source_available():
    source = ROOT / "docs/concepts_explanations.md"
    if not source.exists():
        pytest.skip("Canonical authoring source is not in this checkout")
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
    assert module.resolve_link("../docs/concepts_explanations.md#L2002", aliases, set()) is None
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


@pytest.fixture
def image_build(tmp_path):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    docs = tmp_path / "docs"
    image_dir = docs / "learning_images"
    image_dir.mkdir(parents=True)
    private = PngInfo()
    private.add_text("Comment", "private authoring metadata")
    image = Image.new("RGB", (2000, 1000), "green")
    exif = Image.Exif()
    exif[0x013B] = "private author"
    image.save(image_dir / "own.png", pnginfo=private, exif=exif)
    image.save(image_dir / "unapproved.png")
    source = docs / "concepts.md"
    source.write_text('## Example ^example\n\n![A labelled example](learning_images/own.png)\n')
    entry = {"src": "learning_images/own.png", "license": "own", "concept": "example"}
    manifest = {"concepts": "docs/concepts.md", "diagrams": [], "images": [entry]}
    manifest_path = tmp_path / "content.json"

    def run():
        manifest_path.write_text(json.dumps(manifest))
        return builder().build(tmp_path / "site", tmp_path, manifest_path)

    return tmp_path, source, manifest, run


def test_image_build_strips_metadata_limits_dimensions_and_removes_revoked_images(image_build):
    from PIL import Image

    root, source, manifest, run = image_build
    summary = run()
    files = list((root / "site/concepts/img").glob("*"))
    assert len(files) == 1
    with Image.open(files[0]) as image:
        assert image.size == (1600, 800)
        assert not image.getexif()
        assert "Comment" not in image.info
    html = (root / "site/concepts/index.html").read_text()
    assert summary["images"][0]["url"] in html
    assert 'alt="A labelled example"' in html
    assert "Original diagram (own)" in html
    assert "unapproved.png" not in html
    manifest["images"] = []
    source.write_text("## Example ^example\n\nNo image.\n")
    run()
    assert not list((root / "site/concepts/img").glob("*"))


@pytest.mark.parametrize("license_name", ["course", "unknown", "copyright", ""])
def test_image_build_refuses_unapproved_licenses(image_build, license_name):
    _, _, manifest, run = image_build
    manifest["images"][0]["license"] = license_name
    with pytest.raises(ValueError, match="license"):
        run()


def test_image_build_rejects_unlisted_or_wrong_concept_images(image_build):
    _, source, manifest, run = image_build
    source.write_text('## Example ^example\n\n![secret](learning_images/unapproved.png)\n')
    with pytest.raises(ValueError, match="not allowlisted"):
        run()
    source.write_text('## Example ^example\n\n## Other ^other\n\n![moved](learning_images/own.png)\n')
    with pytest.raises(ValueError, match="not allowlisted"):
        run()


@pytest.mark.parametrize("src", ["../private.png", "/tmp/private.png", "learning_images/../private.png",
                                 "https://example.com/private.png", "learning_images/./own.png"])
def test_image_build_rejects_unsafe_source_paths(image_build, src):
    _, _, manifest, run = image_build
    manifest["images"][0]["src"] = src
    with pytest.raises(ValueError, match="relative learning_images"):
        run()


def test_cc_by_requires_visible_credit_and_source(image_build):
    root, _, manifest, run = image_build
    entry = manifest["images"][0]
    entry["license"] = "cc-by"
    with pytest.raises(ValueError, match="attribution"):
        run()
    entry.update(attribution="Example Author, CC BY 4.0", source_url="https://example.com/diagram")
    run()
    html = (root / "site/concepts/index.html").read_text()
    assert "Example Author, CC BY 4.0" in html
    assert 'href="https://example.com/diagram"' in html
    entry["source_url"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="HTTP URL"):
        run()


def test_image_build_rejects_symlinks_outside_image_directory(image_build):
    root, _, _, run = image_build
    image = root / "docs/learning_images/own.png"
    image.rename(root / "private.png")
    image.symlink_to(root / "private.png")
    with pytest.raises(ValueError, match="escapes"):
        run()


@pytest.mark.parametrize("limit,expected", [("MAX_IMAGE_BYTES", "size limit"),
                                         ("MAX_IMAGE_PIXELS", "megapixel"),
                                         ("MAX_PUBLIC_IMAGE_BYTES", "Public image")])
def test_image_resource_limits_fail_before_publication(image_build, monkeypatch, limit, expected):
    root, _, manifest, _ = image_build
    module = builder()
    monkeypatch.setattr(module, limit, 1)
    with pytest.raises(ValueError, match=expected):
        module.prepare_images(manifest["images"], root, {"example": "example"})
    assert not (root / "site").exists()
