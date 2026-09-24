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
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "apps/learning-content"


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


def rewrite_tokens(tokens, aliases, diagrams):
    stack = []
    for token in tokens:
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
            # The concept source uses text and diagram links, not remote images.
            token.type, token.tag = "text", ""
            token.attrs = {}
        if token.children:
            rewrite_tokens(token.children, aliases, diagrams)


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
    rewrite_tokens(tokens, aliases, diagrams)
    starts = [h for h in headings if h["level"] == 2]
    topics = []
    for index, heading in enumerate(starts):
        end = starts[index + 1]["token_index"] if index + 1 < len(starts) else len(tokens)
        topics.append({**heading, "html": md.renderer.render(tokens[heading["token_index"]:end], md.options, {})})
    env = Environment(loader=FileSystemLoader(SOURCE / "templates"), autoescape=select_autoescape(["html"]))
    for directory in ("concepts", "diagrams/scenes", "diagrams/svg"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "concepts/index.html").write_text(env.get_template("concepts.html").render(topics=topics, headings=headings), encoding="utf-8")
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
        catalog.append({"id": name, "title": title, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "related": related})
    (output / "diagrams/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "diagrams/index.html").write_text(env.get_template("diagrams.html").render(diagrams=catalog), encoding="utf-8")
    summary = {"concept_source": manifest["concepts"], "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
               "topics": len(topics), "headings": len(headings), "diagrams": len(catalog),
               "anchors": aliases, "unpublished_links": "Source references remain as text; private documents are not served."}
    (output / "concepts/build.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in (".html", ".json", ".excalidraw"):
            text = path.read_text()
            if any(value in text for value in ("vscode://", "obsidian://", "/home/", "/opt/projects/")):
                raise ValueError(f"Local-only link or path remains in {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "site")
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps({k: summary[k] for k in ("topics", "headings", "diagrams", "source_sha256")}, indent=2))
