# Import Path to compare image paths loaded from the manifest.
from pathlib import Path


# Test that the sample manifest loads into typed records.
def test_load_sample_galaxy_manifest(load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the tiny sample manifest.
    records = loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # The sample manifest currently contains four small metadata rows.
    assert len(records) == 4

    # Verify that the first row becomes a structured record.
    assert records[0] == loader.GalaxyManifestRecord(
        image_id="gz2-000001",
        image_path=Path("processed/images_224/gz2-000001.jpg"),
        label="spiral",
        split="train",
        source="galaxy_zoo_2",
    )


# Test that records can be grouped into train/val/test buckets.
def test_group_galaxy_manifest_records_by_split(load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_grouped",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the sample manifest records.
    records = loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # Group the records by split.
    grouped = loader.records_by_split(records)

    # Verify the sample split counts.
    assert len(grouped["train"]) == 2
    assert len(grouped["val"]) == 1
    assert len(grouped["test"]) == 1


# Test that relative manifest image paths resolve against a galaxy data root.
def test_resolve_relative_galaxy_image_path(load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_resolve_relative",
        "scripts/load_galaxy_manifest.py",
    )

    # Build a small record with a relative image path like the sample manifest uses.
    record = loader.GalaxyManifestRecord(
        image_id="gz2-000001",
        image_path=Path("processed/images_224/gz2-000001.jpg"),
        label="spiral",
        split="train",
        source="galaxy_zoo_2",
    )

    # Resolve the image path against a fake local root; no real image is required.
    resolved_path = loader.resolve_image_path(
        record,
        Path("/tmp/cosmosai-data/galaxy"),
    )

    # The resolved path should combine the root and the manifest-relative path.
    assert resolved_path == Path(
        "/tmp/cosmosai-data/galaxy/processed/images_224/gz2-000001.jpg"
    )


# Test that absolute image paths are left unchanged by the resolver.
def test_resolve_absolute_galaxy_image_path(load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_resolve_absolute",
        "scripts/load_galaxy_manifest.py",
    )

    # Build a record with an absolute image path.
    record = loader.GalaxyManifestRecord(
        image_id="external-001",
        image_path=Path("/tmp/external-galaxy.jpg"),
        label="spiral",
        split="train",
        source="manual",
    )

    # Resolve the image path against a fake root.
    resolved_path = loader.resolve_image_path(
        record,
        Path("/tmp/cosmosai-data/galaxy"),
    )

    # Absolute paths should not be changed.
    assert resolved_path == Path("/tmp/external-galaxy.jpg")


# Test that loaded records can be converted into an image_id -> full path mapping.
def test_resolved_galaxy_image_paths_by_id(load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_resolved_mapping",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the sample manifest records.
    records = loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # Resolve all image paths against a fake root; real images are not needed yet.
    paths_by_id = loader.resolved_image_paths_by_id(
        records,
        Path("/tmp/cosmosai-data/galaxy"),
    )

    # Verify one known sample record resolves predictably.
    assert paths_by_id["gz2-000001"] == Path(
        "/tmp/cosmosai-data/galaxy/processed/images_224/gz2-000001.jpg"
    )


# Test that missing image paths can be reported without requiring real data.
def test_missing_galaxy_image_paths_with_fake_root(load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_missing_paths",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the sample manifest records.
    records = loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # Use a fake root so every sample image path is missing.
    missing_paths = loader.missing_image_paths(
        records,
        Path("/tmp/cosmosai-data/galaxy"),
    )

    # All four sample records should be reported missing under this fake root.
    assert len(missing_paths) == 4


# Test that existing image files are not reported as missing.
def test_missing_galaxy_image_paths_ignores_existing_file(
    tmp_path,
    load_service_module,
):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_existing_path",
        "scripts/load_galaxy_manifest.py",
    )

    # Load the sample manifest records.
    records = loader.load_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # Create one fake image file matching the first sample manifest row.
    existing_image_path = (
        tmp_path / "processed" / "images_224" / "gz2-000001.jpg"
    )
    existing_image_path.parent.mkdir(parents=True)
    existing_image_path.write_text("fake image placeholder", encoding="utf-8")

    # Check missing paths against the temporary root.
    missing_paths = loader.missing_image_paths(records, tmp_path)

    # The first image exists, so only the remaining three should be missing.
    assert len(missing_paths) == 3
    assert "gz2-000001" not in missing_paths


# Test that invalid manifests are rejected before loading records.
def test_load_manifest_rejects_invalid_manifest(tmp_path, load_service_module):
    # Load the manifest loader script as a module without making scripts/ a package.
    loader = load_service_module(
        "galaxy_manifest_loader_invalid",
        "scripts/load_galaxy_manifest.py",
    )

    # Create a temporary manifest with an invalid label.
    manifest_path = tmp_path / "bad_manifest.csv"
    manifest_path.write_text(
        "image_id,image_path,label,split,source\n"
        "bad-001,processed/images_224/bad-001.jpg,banana,train,galaxy_zoo_2\n",
        encoding="utf-8",
    )

    # Loading should fail because the validator rejects the manifest first.
    try:
        loader.load_manifest(manifest_path)
    except ValueError as error:
        assert "invalid label 'banana'" in str(error)
    else:
        raise AssertionError("Expected invalid manifest to raise ValueError")
