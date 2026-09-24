# Learning Content Build

The gateway serves the generated `site/` tree, not the private `docs/` folder.
Keep generated exports with the app when sharing or building a release. A normal
local launch and Docker build need neither Node nor the private authoring notes.

## Authoring

`content.json` explicitly lists the concept source and the 28 published diagrams.
Edit the canonical Markdown and original `.excalidraw` files, then rebuild both
stages from the repository root:

```bash
.venv/bin/python -m pip install -r requirements-learning.txt
npm ci --prefix tools/learning-site
tools/learning-site/node_modules/.bin/playwright install chromium
.venv/bin/python scripts/build_learning_site.py
node scripts/export_learning_diagrams.cjs
```

The Markdown stage retains all 67 topics and 139 section headings. Stable
Obsidian block IDs become HTML anchors; supported wiki-links, Markdown links and
diagram links become browser URLs. Raw HTML is disabled. Workstation paths and
unpublished file links are not exposed: unavailable references remain text.
Historical examples remain historical, with an explicit note on the page.

The diagram stage uses Excalidraw's official SVG exporter and embeds fonts. The
28 originals stay untouched. Downloads are sanitized scene copies, with web
links replacing local targets; unrelated editor state is omitted. Source hashes
in the catalog/build metadata allow tests to detect stale published content.
Run the SVG stage after the Markdown stage, which resets the catalog.

New image attachments are rejected pending a publication review. Adding a
diagram requires an explicit allowlist change and a rebuild. No runtime Markdown
parser, editor, CDN or remote font service is needed by the browser.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_learning_content.py tests/test_local_classify_demo.py
# With the local demo running:
node tools/learning-site/check-browser.cjs http://127.0.0.1:8080
```

The browser check covers an imported trace round-trip, concept search, diagram
pixels, four viewport widths and standalone replay compatibility. Screenshots
default to `/tmp/cosmosai-learning-check` (`COSMOSAI_SCREENSHOTS` overrides it).
`COSMOSAI_LEARNING_TOOLS` may point to a separate directory containing the same
Node dependencies; `PLAYWRIGHT_BROWSERS_PATH` is also supported.

Replay state is stored only in the current browser tab's `sessionStorage`, not
sent to services. A disabled/full store leaves normal navigation available;
an unavailable return snapshot is explicitly reported with the bundled trace.
Font attribution is in `apps/api-gateway/static/vendor/learning-licenses/`.
