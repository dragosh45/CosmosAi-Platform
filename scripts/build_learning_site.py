#!/usr/bin/env python3
"""Publish the allowlisted concepts/diagrams, never the whole private docs tree.

Markdown is parsed into tokens (including a small Obsidian wiki-link extension).
Stable heading/block IDs are shared by the replay, concepts and scene exports.
Raw HTML is disabled. Unpublished source references remain readable, unlinked.
Run export_learning_diagrams.cjs afterward to render the sanitized scenes to SVG.
"""

import argparse
from collections import Counter
import hashlib
from html import escape
import io
import json
from pathlib import Path
import re
import shutil
from urllib.parse import parse_qs, unquote, urlsplit
import xml.etree.ElementTree as ET

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "apps/learning-content"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_PUBLIC_IMAGE_BYTES = 1024 * 1024
EXAMPLES = {
    "cnn-fundamentals": {"title": "CNN: sliding filters, shapes and parameter counts", "url": "/learn/examples/cnn-fundamentals/"},
    "lab1-preprocessing": {"title": "Lab 1: interpolation and normalisation with numbers", "url": "/learn/examples/lab1-preprocessing/"},
    "weighted-sum": {"title": "One neuron: weights, bias and ReLU", "url": "/learn/examples/weighted-sum/"},
    "cnn-replay-weights": {"title": "CNN Replay: inspect the exact weights", "url": "/learn/#parameter-inventory"},
    "cnn-replay-loss": {"title": "CNN Replay: follow the loss calculation", "url": "/learn/#loss-walkthrough"},
}


def load_courses(manifest, source_root, headings, aliases):
    """Project only explicit public metadata; never export notes or PDF paths."""
    if not manifest.get("course_map"):
        return []
    # JSON is a subset of YAML; this small authoring map needs no YAML runtime.
    raw = json.loads((source_root / manifest["course_map"]).read_text())
    titles = {heading["id"]: heading["title"] for heading in headings}
    owners = {heading["id"]: heading["topic"] for heading in headings}
    courses, seen = [], set()
    for entry in raw["courses"]:
        course_id = entry["id"]
        if not re.fullmatch(r"[A-Z][A-Z0-9_-]*", course_id) or course_id in seen:
            raise ValueError("Invalid or duplicate course ID")
        seen.add(course_id)
        lectures, lecture_ids, topics = [], set(), set()
        for lesson in entry["lectures"]:
            if lesson["id"] in lecture_ids or not re.fullmatch(r"[A-Z0-9_-]+", lesson["id"]):
                raise ValueError("Invalid or duplicate lecture ID")
            lecture_ids.add(lesson["id"])
            concepts = []
            for key in lesson.get("concepts", []):
                anchor = aliases.get(key)
                if anchor not in titles:
                    raise ValueError(f"Unknown course concept: {key}")
                concepts.append({"id": anchor, "title": titles[anchor]})
                topics.add(owners[anchor])
            examples = []
            for key in lesson.get("examples", []):
                if key not in EXAMPLES:
                    raise ValueError(f"Unknown worked example: {key}")
                examples.append({"id": key, **EXAMPLES[key]})
            lectures.append({"id": lesson["id"], "title": lesson["title"],
                             "concepts": concepts, "examples": examples})
        courses.append({"id": course_id, "title": entry["title"], "lectures": lectures,
                        "topics": sorted(topics)})
    return courses


def write_courses(output, courses, env, private_study=False):
    directory = output / "courses"
    # Remove retired course pages, including old local-only attachments.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    (directory / "index.html").write_text(env.get_template("courses.html").render(
        courses=courses, private_study=private_study), encoding="utf-8")
    for course in courses:
        destination = directory / course["id"]
        destination.mkdir()
        (destination / "index.html").write_text(env.get_template("course.html").render(
            course=course, private_study=private_study), encoding="utf-8")
    (output / "learning_map.json").write_text(json.dumps({"courses": courses}, indent=2) + "\n")


def prepare_images(entries, source_root, aliases):
    """Validate reviewed local images and re-encode pixels without metadata."""
    images = {}
    image_root = (source_root / "docs/learning_images").resolve()
    for entry in entries:
        src = entry["src"]
        relative = Path(src)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in src
                or relative.as_posix() != src or not src.startswith("learning_images/")):
            raise ValueError("Image source must be a relative learning_images/ path")
        path = (source_root / "docs" / relative).resolve()
        if not path.is_relative_to(image_root) or src in images:
            raise ValueError("Image source escapes learning_images/ or is duplicated")
        license_name = entry.get("license")
        if license_name not in {"own", "cc-by", "public-domain"}:
            raise ValueError("Image license must be own, cc-by or public-domain")
        credit = entry.get("attribution", "").strip()
        source_url = entry.get("source_url", "")
        valid_source = urlsplit(source_url).scheme in {"http", "https"} and bool(urlsplit(source_url).netloc)
        if source_url and not valid_source:
            raise ValueError("Image source_url must be an HTTP URL")
        if license_name == "cc-by" and (not credit or not valid_source):
            raise ValueError("CC BY images need attribution and an HTTP source_url")
        concept = aliases.get(entry["concept"])
        if concept is None:
            raise ValueError("Image concept must name an existing concept anchor")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Image source exceeds the 10 MiB size limit")
        with Image.open(path) as original:
            if original.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Only PNG, JPEG and WebP image sources are supported")
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit")
            oriented = ImageOps.exif_transpose(original)
            oriented.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            # A new image carries pixels only: no EXIF, comments, ICC or PNG text.
            converted = oriented.convert("RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB")
            clean = Image.new(converted.mode, converted.size)
            clean.paste(converted)
            buffer = io.BytesIO()
            clean.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        if len(data) > MAX_PUBLIC_IMAGE_BYTES:
            raise ValueError("Public image exceeds 1 MiB; simplify or resize it before review")
        name = hashlib.sha256(src.encode()).hexdigest()[:16] + ".png"
        images[src] = {"src": src, "license": license_name, "concept": concept,
                       "attribution": credit, "source_url": source_url,
                       "url": "/learn/concepts/img/" + name, "name": name,
                       "width": clean.width, "height": clean.height,
                       "sha256": hashlib.sha256(data).hexdigest(), "data": data}
    return images


def render_image(self, tokens, index, options, env):
    """Show ownership/attribution alongside each approved image."""
    token = tokens[index]
    item = token.meta["approved_image"]
    credit = item["attribution"] or {"own": "Original diagram", "public-domain": "Public domain"}.get(item["license"], "")
    if item["source_url"]:
        credit = '<a href="' + escape(item["source_url"], quote=True) + '" rel="noopener noreferrer">' + escape(credit) + '</a>'
    else:
        credit = escape(credit)
    alt = escape(self.renderInlineAsText(token.children or [], options, env), quote=True)
    return (f'<span class="concept-figure"><img src="{item["url"]}" alt="{alt}" '
            f'width="{item["width"]}" height="{item["height"]}" loading="lazy" decoding="async">'
            f'<span class="image-credit">{credit} ({escape(item["license"])})</span></span>')


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def public_text(text):
    """Keep project-relative examples while removing workstation directory names."""
    return re.sub(r"/(?:home/[^/]+|opt/projects)/cosmosai-platform(/)?",
                  lambda match: "" if match[1] else "the project root", text)


def wiki_link(state, silent):
    if not state.src.startswith("[[", state.pos):
        return False
    end = state.src.find("]]", state.pos + 2)
    if end < 0:
        return False
    target, _, label = state.src[state.pos + 2:end].partition("|")
    if not silent:
        token = state.push("link_open", "a", 1)
        token.attrSet("href", target)
        state.push("text", "", 0).content = label or target.lstrip("#^")
        state.push("link_close", "a", -1)
    state.pos = end + 2
    return True


def parse_concepts(source):
    md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
    md.add_render_rule("image", render_image)
    md.inline.ruler.before("link", "wiki_link", wiki_link)
    tokens = md.parse(public_text(source))
    headings, aliases, counts = [], {}, Counter()
    owner = None
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        inline = tokens[index + 1]
        match = re.search(r"\s+\^([\w-]+)$", inline.content)
        title = inline.content[:match.start()] if match else inline.content
        base = match[1] if match else slug(title)
        counts[base] += 1
        anchor = base if counts[base] == 1 else f"{base}-{counts[base]}"
        token.attrSet("id", anchor)
        inline.content = title
        inline.children = md.parseInline(title)[0].children
        if token.tag == "h2":
            owner = anchor
        if token.tag != "h1":
            headings.append({"id": anchor, "title": title, "level": int(token.tag[1]),
                             "topic": owner, "token_index": index})
        for alias in (anchor, slug(title), title, "^" + base):
            aliases.setdefault(alias, anchor)
    return md, tokens, headings, aliases


def resolve_link(target, aliases, diagrams):
    if target.startswith("obsidian://"):
        parsed = urlsplit(target)
        target = parse_qs(parsed.query).get("path", [""])[0]
        if parsed.fragment:
            target += "#" + parsed.fragment
    target = unquote(target)
    parsed = urlsplit(target)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return target
    path, _, fragment = target.partition("#")
    if re.fullmatch(r"L\d+(?:-L\d+)?", fragment):
        return None
    if not path or Path(path).name in ("concepts_explanations", "concepts_explanations.md"):
        key = fragment.lstrip("^")
        anchor = aliases.get(key) or aliases.get(fragment) or aliases.get(slug(key))
        if anchor:
            return "/learn/concepts/#" + anchor
        if not fragment:
            return "/learn/concepts/"
        raise ValueError(f"Unknown concept anchor: {fragment}")
    if parsed.scheme:
        return None
    diagram = Path(path).stem
    if diagram in diagrams:
        return "/learn/diagrams/#" + diagram
    return None


def rewrite_tokens(tokens, aliases, diagrams, images=None, topic=None):
    stack = []
    for token in tokens:
        if token.type == "heading_open" and token.tag == "h2":
            topic = token.attrGet("id")
        if token.type == "link_open":
            href = resolve_link(token.attrGet("href"), aliases, diagrams)
            stack.append(bool(href))
            if href:
                token.attrSet("href", href)
                if href.startswith("https://") or href.startswith("http://"):
                    token.attrSet("rel", "noopener noreferrer")
            else:
                token.tag = "span"
                token.attrs = {"class": "source-reference"}
        elif token.type == "link_close" and stack:
            if not stack.pop():
                token.tag = "span"
        elif token.type == "image":
            item = (images or {}).get(token.attrGet("src"))
            if item is None or item["concept"] != topic:
                raise ValueError("Image is not allowlisted for this concept; review it before publication")
            token.meta["approved_image"] = item
            token.attrSet("src", item["url"])
        if token.children:
            rewrite_tokens(token.children, aliases, diagrams, images, topic)


def sanitize_scene(scene, aliases, diagrams):
    """Copy only rendering data; remove local/unsupported targets and app state."""
    elements = []
    for original in scene["elements"]:
        if original.get("isDeleted"):
            continue
        element = dict(original)
        for key in ("text", "originalText", "rawText"):
            if isinstance(element.get(key), str):
                element[key] = public_text(element[key])
        element["link"] = resolve_link(element["link"], aliases, diagrams) if element.get("link") else None
        elements.append(element)
    if scene.get("files"):
        raise ValueError("Embedded diagram files need review before publication")
    return {"type": "excalidraw", "version": 2, "source": "CosmosAI learning export",
            "elements": elements, "appState": {"viewBackgroundColor": "#ffffff"}, "files": {}}


def build(output=SOURCE / "site", source_root=ROOT, manifest_path=SOURCE / "content.json"):
    manifest = json.loads(manifest_path.read_text())
    source_path = source_root / manifest["concepts"]
    source = source_path.read_text(encoding="utf-8")
    md, tokens, headings, aliases = parse_concepts(source)
    diagrams = set(manifest["diagrams"])
    images = prepare_images(manifest.get("images", []), source_root, aliases)
    rewrite_tokens(tokens, aliases, diagrams, images)
    starts = [h for h in headings if h["level"] == 2]
    topics = []
    concept_examples = manifest.get("concept_examples", {})
    for anchor, examples in concept_examples.items():
        if anchor not in aliases or any(key not in EXAMPLES for key in examples):
            raise ValueError("Unknown concept or worked example in concept_examples")
    for index, heading in enumerate(starts):
        end = starts[index + 1]["token_index"] if index + 1 < len(starts) else len(tokens)
        topics.append({**heading, "html": md.renderer.render(tokens[heading["token_index"]:end], md.options, {}),
                       "examples": [EXAMPLES[key] for key in concept_examples.get(heading["id"], [])]})
    env = Environment(loader=FileSystemLoader(SOURCE / "templates"), autoescape=select_autoescape(["html"]))
    courses = load_courses(manifest, source_root, headings, aliases)
    for directory in ("concepts", "diagrams/scenes", "diagrams/svg"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    # Rebuild this generated directory so removing approval also removes old pixels.
    image_output = output / "concepts/img"
    if image_output.exists():
        shutil.rmtree(image_output)
    if images:
        image_output.mkdir()
        for item in images.values():
            (image_output / item["name"]).write_bytes(item["data"])
    (output / "concepts/index.html").write_text(env.get_template("concepts.html").render(
        topics=topics, headings=headings, courses=courses), encoding="utf-8")
    write_courses(output, courses, env)
    example_source = ROOT / "apps/learning-examples"
    if example_source.exists():
        if (output / "examples").exists():
            shutil.rmtree(output / "examples")
        shutil.copytree(example_source, output / "examples")
    catalog = []
    for name in manifest["diagrams"]:
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError("Invalid diagram name in allowlist")
        path = source_root / "docs/excalidraw" / (name + ".excalidraw")
        original = json.loads(path.read_text())
        scene = sanitize_scene(original, aliases, diagrams)
        # Original diagram titles are more informative than machine filenames.
        texts = sorted((e for e in scene["elements"] if e["type"] == "text"), key=lambda e: (e["y"], e["x"]))
        title = texts[0]["text"].split("\n")[0] if texts else name.replace("_", " ").title()
        related = sorted({e["link"] for e in scene["elements"] if (e.get("link") or "").startswith("/learn/concepts/#")})
        (output / "diagrams/scenes" / (name + ".excalidraw")).write_text(json.dumps(scene, ensure_ascii=True), encoding="utf-8")
        item = {"id": name, "title": title,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "related": related}
        # A concepts-only rebuild can retain dimensions from an existing SVG.
        # The exporter still owns creating or changing diagram renderings.
        svg_path = output / "diagrams/svg" / (name + ".svg")
        if svg_path.exists():
            viewbox = ET.fromstring(svg_path.read_text()).attrib.get("viewBox", "").split()
            if len(viewbox) == 4:
                dimensions = [float(value) for value in viewbox[2:]]
                item["width"], item["height"] = [
                    int(value) if value.is_integer() else value for value in dimensions
                ]
        catalog.append(item)
    (output / "diagrams/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "diagrams/index.html").write_text(env.get_template("diagrams.html").render(diagrams=catalog), encoding="utf-8")
    summary = {"concept_source": manifest["concepts"], "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
               "topics": len(topics), "headings": len(headings), "diagrams": len(catalog),
               "anchors": aliases, "unpublished_links": "Source references remain as text; private documents are not served."}
    summary["images"] = [{k: v for k, v in item.items() if k not in {"data", "name"}} for item in images.values()]
    (output / "concepts/build.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in (".html", ".json", ".excalidraw"):
            text = path.read_text()
            if any(value in text for value in ("vscode://", "obsidian://", "/home/", "/Users/", "/opt/projects/")):
                raise ValueError(f"Local-only link or path remains in {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "site")
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps({k: summary[k] for k in ("topics", "headings", "diagrams", "source_sha256")}, indent=2))
