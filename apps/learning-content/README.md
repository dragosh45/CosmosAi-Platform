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

The Markdown stage retains every reviewed topic and section heading. Stable
Obsidian block IDs become HTML anchors; supported wiki-links, Markdown links and
diagram links become browser URLs. Raw HTML is disabled. Workstation paths and
unpublished file links are not exposed: unavailable references remain text.
Historical examples remain historical, with an explicit note on the page.

The diagram stage uses Excalidraw's official SVG exporter and embeds fonts. The
28 originals stay untouched. Downloads are sanitized scene copies, with web
links replacing local targets; unrelated editor state is omitted. Source hashes
in the catalog/build metadata allow tests to detect stale published content.
Run the SVG stage after the Markdown stage, which resets the catalog.

Concept images require an explicit entry in the `images` allowlist and a normal
Markdown image reference using the same `src` beneath its approved concept:

```json
{"src": "learning_images/example.png", "license": "own", "concept": "example-anchor"}
```

Paths resolve under the private authoring folder. Only `own`, `cc-by` and
`public-domain` are accepted; CC BY entries also require `attribution` (including
the license version) and `source_url`. Credits are shown beside each image.
Unlisted images, remote images, path escapes and use under another concept fail
the build. Source PNG/JPEG/WebP files must be at most 10 MiB and 16 megapixels.
The build fits them within 1600 × 1600 pixels, strips metadata, and writes PNGs
of at most 1 MiB into `site/concepts/img/`. This directory is regenerated, so
removing approval also removes a previous image. Build metadata records each
image's license, concept, dimensions and output checksum. The default image list
is empty until reviewed images are approved.

Adding a diagram requires an explicit allowlist change and a rebuild. No runtime
Markdown parser, editor, CDN or remote font service is needed by the browser.

## Verification

`/learn/courses/` groups lessons, existing concept anchors and practical examples.
The course selector on Concepts filters the navigation without duplicating the
explanations. The authoring map uses JSON-compatible YAML; the generated
`site/learning_map.json` contains only explicitly selected course metadata and
links. Notes, PDF paths and unknown authoring fields are omitted.

The one-neuron example at `/learn/examples/weighted-sum/` accepts numeric inputs,
weights and bias and shows products, the sum and ReLU. Reproduce its default JSON
with `python scripts/examples/weighted_sum.py` before rebuilding. It calculates
in the browser without training or AI requests.

To serve only learning pages and examples, without loading model services:

```bash
python scripts/run_local_learning.py --port 8080
```

`--content-root` accepts a separately generated local preview tree. The launcher
binds only to localhost. It does not mount authoring folders or start inference.

```bash
.venv/bin/python -m pytest -q tests/test_learning_content.py tests/test_course_learning.py tests/test_repository_portability.py
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

## Lab 1 arithmetic practice

`/learn/examples/lab1-preprocessing/` shows linear interpolation and z-score
normalisation with editable inputs, visible substitutions and an explanation of
standard deviation. Reproduce its default trace with:

```bash
python scripts/examples/lab1_preprocessing.py
python scripts/build_learning_site.py
```

The AIA course has a Lab 1 entry. The separate local study preview adds eight
lab concept drafts, seven MATLAB-result figures, and explicit brief/report/script
copies. These new explanations are not merged into canonical Concepts yet.
The optional local-preview browser check is
`node tools/learning-site/check-lab1-browser.cjs`; it checks desktop/mobile layout,
calculations, image loading and source-file checksums. No AI calls or MATLAB
execution are involved in the arithmetic example.


### CNN calculation practice

`/learn/examples/cnn-fundamentals/` adds a bounded browser exercise alongside
Replay: sliding two-channel cross-correlation, per-channel products and bias,
ReLU, global pooling, output-size and shared-parameter calculations. It explains
the existing 132-parameter TinyGalaxyCNN and includes self-check questions.
These hand-chosen fixtures are separate from Replay's saved model values.
Run arithmetic checks with `node --test tests/test_cnn_fundamentals.cjs`.

The CNN example also explains kernel weight slices and matching-position
multiplication, with a stride/stacking SVG and layer/stride receptive-field
controls. Replay's weight-origin panel uses the recorded classifier parameters
and gradients to distinguish pooled activations from stored connection weights.

## Check a packaged release

Run `python scripts/check_release_demo.py --base-url http://127.0.0.1:8090`
after starting the release Compose stack. It checks the learning pages and
example assets against this checkout, then verifies inference and upload errors.
Run `node tools/learning-site/check-browser.cjs http://127.0.0.1:8090` for course
navigation, arithmetic controls, receptive fields, recorded weight updates,
diagram rendering and responsive layout. Set `COSMOSAI_CHROME` to an installed
Chrome executable if you do not use Playwright's downloaded Chromium.

The generated course index includes AIA, AIMLA and MAL. Source attachments and
separate local study overlays are not included by the standard build.
