"""Repository context stays portable without absorbing large local artifacts."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
MAX_REPOSITORY_ARTIFACT_BYTES = 1024 * 1024
MODEL_ARTIFACTS = (
    ROOT / "models/tiny_galaxy_cnn_baseline.pt",
    ROOT / "models/gz2-validation-30epochs-64px/candidate.pt",
    ROOT / "models/gz2-validation-30epochs-64px/evaluation.json",
    ROOT / "models/gz2-validation-30epochs-64px/model_card.json",
    ROOT / "models/gz2-learning-check-adam/candidate.pt",
    ROOT / "models/gz2-learning-check-adam/evaluation.json",
    ROOT / "models/gz2-learning-check-adam/model_card.json",
)
MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_context_has_no_large_files():
    # docs/ is a separate private repository (ignored here), so only tracked folders are checked.
    paths = [path for path in (ROOT / "data").rglob("*") if path.is_file()]
    paths.extend(MODEL_ARTIFACTS)

    oversized = {str(path.relative_to(ROOT)): path.stat().st_size
                 for path in paths if path.stat().st_size > MAX_REPOSITORY_ARTIFACT_BYTES}
    assert not oversized


def test_candidate_model_cards_match_their_checkpoints():
    for directory in ("gz2-validation-30epochs-64px", "gz2-learning-check-adam"):
        root = ROOT / "models" / directory
        model_card = json.loads((root / "model_card.json").read_text())
        assert model_card["checkpoint_sha256"] == sha256(root / "candidate.pt")


def test_cross_machine_context_entry_points_exist():
    paths = [ROOT / "AGENTS.md", ROOT / "data/README.md", ROOT / "models/README.md"]
    # docs/ is the maintainer's private context repository; check it only when cloned.
    if (ROOT / "docs").is_dir():
        paths.extend(ROOT / name for name in (
            "docs/README.md", "docs/current_status.md", "docs/build_log.md", "docs/workflow.md"))
    for path in paths:
        assert path.is_file(), path


def test_relative_markdown_links_resolve_in_a_fresh_clone():
    documents = [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "deploy/README.md"]
    documents.extend((ROOT / "docs").rglob("*.md"))
    documents.extend((ROOT / "data").rglob("*.md"))
    documents.extend((ROOT / "models").rglob("*.md"))

    missing = []
    for document in documents:
        in_fence = False
        for line in document.read_text().splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            line = re.sub(r"`[^`]*`", "", line)
            for raw_target in MARKDOWN_LINK.findall(line):
                target = raw_target.strip().strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith(("#", "/")) or not parsed.path:
                    continue
                destination = (document.parent / unquote(parsed.path)).resolve()
                if not destination.exists():
                    missing.append((str(document.relative_to(ROOT)), target))

    assert not missing
