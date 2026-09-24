#!/usr/bin/env python3
"""Build a bounded, reproducible Galaxy Zoo 2 pilot for the CosmosAI CNN.

This script reads the official GZ2 Hart et al. metadata and the Zenodo
object-to-image mapping, applies the documented high-confidence four-label
policy, writes a manifest on the configured data drive, and optionally
downloads one SDSS DR7 cutout per selected galaxy. The lenticular label is a
documented derived teaching proxy because GZ2 does not publish it as a
top-level class.

The workflow is intentionally bounded: it never downloads the full image
archive and keeps source files, manifests, reports, and images outside the
Git repository.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image


DEFAULT_ROOT = Path("/media/h1dr0/KINGSTON/cosmosai-data/galaxy")
METADATA_URL = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
MAPPING_URL = "https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1"
IMAGE_SERVICE_URL = "https://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx"
LABELS = ("elliptical", "spiral", "lenticular", "irregular")


def as_float(row: dict[str, str], name: str) -> float:
    """Read a numeric GZ2 vote field; blank fields become 0.0."""
    value = row.get(name, "")
    return float(value) if value else 0.0


def candidate_labels(row: dict[str, str]) -> tuple[str, ...]:
    """Return labels whose conservative GZ2 vote rules pass.

    choose_rows later excludes rows with no matching rule or more than one
    matching rule instead of guessing when the evidence is ambiguous.
    """

    smooth = as_float(row, "t01_smooth_or_features_a01_smooth_debiased")
    features = as_float(row, "t01_smooth_or_features_a02_features_or_disk_debiased")
    spiral = as_float(row, "t04_spiral_a08_spiral_debiased")
    irregular = as_float(row, "t08_odd_feature_a22_irregular_debiased")
    merger = as_float(row, "t08_odd_feature_a24_merger_debiased")
    morphology = row.get("gz2_class", "")
    candidates: list[str] = []
    if morphology.startswith("E") and smooth >= 0.8:
        candidates.append("elliptical")
    if morphology.startswith("S") and spiral >= 0.8:
        candidates.append("spiral")
    if irregular >= 0.8 and merger < 0.2:
        candidates.append("irregular")
    # GZ2 has no official top-level "lenticular" class. This is a documented
    # derived teaching label: disk/features votes, no spiral, irregular or merger.
    if features >= 0.8 and spiral < 0.2 and irregular < 0.2 and merger < 0.2:
        candidates.append("lenticular")
    return tuple(candidates)


def label_confidence(row: dict[str, str], label: str) -> float:
    """Return the weakest vote supporting the selected derived label."""
    values = {
        "elliptical": [as_float(row, "t01_smooth_or_features_a01_smooth_debiased")],
        "spiral": [as_float(row, "t04_spiral_a08_spiral_debiased")],
        "irregular": [as_float(row, "t08_odd_feature_a22_irregular_debiased")],
        "lenticular": [
            as_float(row, "t01_smooth_or_features_a02_features_or_disk_debiased"),
            1.0 - as_float(row, "t04_spiral_a08_spiral_debiased"),
            1.0 - as_float(row, "t08_odd_feature_a22_irregular_debiased"),
            1.0 - as_float(row, "t08_odd_feature_a24_merger_debiased"),
        ],
    }
    return min(values[label])


def load_mapping(path: Path) -> dict[tuple[str, str], str]:
    """Load the object/sample-to-image mapping and reject conflicts."""
    mapping: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["objid"], row["sample"])
            if key in mapping and mapping[key] != row["asset_id"]:
                raise ValueError(f"Conflicting image mapping for {key}")
            mapping[key] = row["asset_id"]
    return mapping


def image_url(row: dict[str, str]) -> str:
    """Build a fixed-size SDSS DR7 cutout URL for one catalog row."""
    query = urlencode(
        {
            "ra": row["ra"],
            "dec": row["dec"],
            "width": 424,
            "height": 424,
        }
    )
    return f"{IMAGE_SERVICE_URL}?{query}"


def choose_rows(metadata_path: Path, mapping_path: Path, per_class: int, seed: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Audit metadata, select deterministic rows, and assign train/val/test splits.

    Only original-sample objects with exactly one high-confidence label rule
    are eligible. A hash of the caller-provided seed and object id makes the
    bounded selection reproducible without relying on global random state.
    """
    mapping = load_mapping(mapping_path)
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    audit = Counter()
    with gzip.open(metadata_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            audit["metadata_rows"] += 1
            if row.get("sample") != "original":
                audit["non_original_rows"] += 1
                continue
            key = (row["dr7objid"], row["sample"])
            if key not in mapping:
                audit["missing_image_mapping"] += 1
                continue
            labels = candidate_labels(row)
            if not labels:
                audit["unassigned"] += 1
                continue
            if len(labels) > 1:
                audit["conflicting_label_rules"] += 1
                continue
            label = labels[0]
            row = dict(row)
            row["label"] = label
            row["confidence"] = f"{label_confidence(row, label):.6f}"
            row["asset_id"] = mapping[key]
            candidates[label].append(row)
            audit[f"candidate_{label}"] += 1

    selected: list[dict[str, str]] = []
    for label in LABELS:
        rows = sorted(
            candidates[label],
            key=lambda row: hashlib.sha256(f"{seed}:{row['dr7objid']}".encode()).hexdigest(),
        )
        if len(rows) < per_class:
            raise ValueError(f"Only {len(rows)} rows available for {label}; need {per_class}")
        for index, row in enumerate(rows[:per_class]):
            split = "train" if index < round(per_class * 0.7) else "val" if index < round(per_class * 0.85) else "test"
            row["split"] = split
            row["image_id"] = f"gz2-{row['dr7objid']}"
            row["image_path"] = f"images/{row['image_id']}.jpg"
            row["source"] = "galaxy_zoo_2"
            row["source_image_url"] = image_url(row)
            row["label_rule"] = "GZ2 high-confidence derived rule v1"
            selected.append(row)
    audit["selected_rows"] = len(selected)
    return sorted(selected, key=lambda row: row["image_id"]), dict(audit)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the CosmosAI manifest columns together with provenance fields."""
    fields = [
        "image_id", "image_path", "label", "split", "source", "dr7objid",
        "sample", "asset_id", "gz2_class", "confidence", "ra", "dec",
        "label_rule", "source_image_url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def validate_image(path: Path) -> tuple[int, int]:
    """Verify that an image decodes successfully and return its dimensions."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def download_one(row: dict[str, str], root: Path, timeout: int) -> tuple[str, str, str]:
    """Download and validate one image, reusing an existing valid file.

    Downloads first go to a temporary .part path and are moved into place
    only after validation so an interrupted transfer cannot look complete.
    """
    target = root / row["image_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        width, height = validate_image(target)
        return row["image_id"], "existing", f"{width}x{height}"
    temporary = target.with_suffix(".jpg.part")
    request = Request(row["source_image_url"], headers={"User-Agent": "CosmosAI bounded dataset intake"})
    with urlopen(request, timeout=timeout) as response:
        temporary.write_bytes(response.read())
    width, height = validate_image(temporary)
    temporary.replace(target)
    time.sleep(0.1)
    return row["image_id"], "downloaded", f"{width}x{height}"


def main() -> int:
    """Run selection, manifest/report creation, and optional image downloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", default="cosmosai-gz2-pilot-v1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    # Keep downloaded source material and generated artifacts on the data drive.
    source = args.root / "source"
    manifests = args.root / "manifests"
    images = args.root / "images"
    manifests.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    metadata_path = source / "gz2_hart16.csv.gz"
    mapping_path = source / "gz2_filename_mapping.csv"
    if not metadata_path.exists() or not mapping_path.exists():
        raise SystemExit("Download gz2_hart16.csv.gz and gz2_filename_mapping.csv into source first.")

    rows, audit = choose_rows(metadata_path, mapping_path, args.per_class, args.seed)
    # Write the selection before downloading so the dataset remains auditable.
    manifest_path = manifests / f"gz2_pilot_{len(rows)}.csv"
    write_manifest(manifest_path, rows)
    report = {
        "source": "Galaxy Zoo 2 Hart et al. 2016 Table 1 + Zenodo original image mapping",
        "metadata_url": METADATA_URL,
        "mapping_url": MAPPING_URL,
        "image_service": IMAGE_SERVICE_URL,
        "selection_seed": args.seed,
        "per_class": args.per_class,
        "label_rule": {
            "elliptical": "gz2_class starts E and smooth debiased vote >= 0.8",
            "spiral": "gz2_class starts S and spiral debiased vote >= 0.8",
            "irregular": "irregular debiased vote >= 0.8 and merger vote < 0.2",
            "lenticular": "features/disk vote >= 0.8; spiral, irregular and merger votes < 0.2; derived proxy, not an official GZ2 class",
        },
        "audit": audit,
        "selected_by_label": dict(Counter(row["label"] for row in rows)),
        "selected_by_split": dict(Counter(row["split"] for row in rows)),
        "manifest": str(manifest_path),
    }
    (manifests / f"gz2_pilot_{len(rows)}.selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.plan_only:
        return 0

    # Download only the bounded, already-selected rows, with limited concurrency.
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, row, args.root, args.timeout): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                image_id, status, size = future.result()
                result = {"image_id": image_id, "status": status, "size": size}
                print(f"{image_id}: {status} {size}")
            except Exception as error:  # noqa: BLE001 - preserve every failed source row in the report.
                # Keep failures in the report so one bad source does not hide the rest.
                result = {"image_id": row["image_id"], "status": "failed", "error": str(error)}
                print(f"{row['image_id']}: failed {error}")
            results.append(result)
    results.sort(key=lambda item: item["image_id"])
    download_report = manifests / f"gz2_pilot_{len(rows)}.download.json"
    download_report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    failed = [item for item in results if item["status"] == "failed"]
    print(f"Downloaded/verified {len(results) - len(failed)} of {len(results)} images")
    if failed:
        print(f"Failures recorded in {download_report}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
