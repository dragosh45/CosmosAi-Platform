# Import Path to create temporary manifest paths in tests.
from pathlib import Path


# Test that the repository sample manifest matches the galaxy data contract.
def test_sample_galaxy_manifest_is_valid(load_service_module):
    # Load the validator script as a module without requiring scripts/ to be a package.
    validator = load_service_module(
        "galaxy_manifest_validator",
        "scripts/validate_galaxy_manifest.py",
    )

    # Validate the tiny sample manifest committed with the project.
    errors = validator.validate_manifest(
        Path("data/samples/galaxy_manifest_sample.csv")
    )

    # A valid manifest returns no errors.
    assert errors == []


# Test that invalid labels are rejected with a useful error.
def test_galaxy_manifest_rejects_invalid_label(tmp_path, load_service_module):
    # Load the validator script as a module without requiring scripts/ to be a package.
    validator = load_service_module(
        "galaxy_manifest_validator_invalid_label",
        "scripts/validate_galaxy_manifest.py",
    )

    # Create a temporary manifest with one invalid galaxy label.
    manifest_path = tmp_path / "bad_manifest.csv"
    manifest_path.write_text(
        "image_id,image_path,label,split,source\n"
        "bad-001,processed/images_224/bad-001.jpg,banana,train,galaxy_zoo_2\n",
        encoding="utf-8",
    )

    # Validate the temporary manifest.
    errors = validator.validate_manifest(manifest_path)

    # The invalid label should be reported.
    assert any("invalid label 'banana'" in error for error in errors)


# Test that invalid split values are rejected with a useful error.
def test_galaxy_manifest_rejects_invalid_split(tmp_path, load_service_module):
    # Load the validator script as a module without requiring scripts/ to be a package.
    validator = load_service_module(
        "galaxy_manifest_validator_invalid_split",
        "scripts/validate_galaxy_manifest.py",
    )

    # Create a temporary manifest with one invalid split value.
    manifest_path = tmp_path / "bad_manifest.csv"
    manifest_path.write_text(
        "image_id,image_path,label,split,source\n"
        "bad-002,processed/images_224/bad-002.jpg,spiral,production,galaxy_zoo_2\n",
        encoding="utf-8",
    )

    # Validate the temporary manifest.
    errors = validator.validate_manifest(manifest_path)

    # The invalid split should be reported.
    assert any("invalid split 'production'" in error for error in errors)
