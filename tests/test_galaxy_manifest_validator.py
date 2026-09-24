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


# Test that a short CSV row names the exact missing field and source row.
def test_galaxy_manifest_reports_field_missing_from_short_row(
    tmp_path,
    load_service_module,
):
    validator = load_service_module(
        "galaxy_manifest_validator_short_row",
        "scripts/validate_galaxy_manifest.py",
    )
    manifest_path = tmp_path / "short_row.csv"
    manifest_path.write_text(
        "image_id,image_path,label,split,source\n"
        "bad-003,processed/bad-003.jpg,spiral,train\n",
        encoding="utf-8",
    )

    errors = validator.validate_manifest(manifest_path)

    assert errors == [
        "Row 2, field 'source': value is missing because the row has fewer "
        "columns than the header"
    ]


# Test that extra CSV values are rejected instead of silently stored under None.
def test_galaxy_manifest_reports_extra_row_values(tmp_path, load_service_module):
    validator = load_service_module(
        "galaxy_manifest_validator_extra_value",
        "scripts/validate_galaxy_manifest.py",
    )
    manifest_path = tmp_path / "extra_value.csv"
    manifest_path.write_text(
        "image_id,image_path,label,split,source\n"
        "bad-004,processed/bad-004.jpg,spiral,train,test,unexpected\n",
        encoding="utf-8",
    )

    errors = validator.validate_manifest(manifest_path)

    assert errors == [
        "Row 2: has extra value(s) beyond the 5 header columns: ['unexpected']"
    ]


# Test that malformed CSV quoting becomes a validation error, not a traceback.
def test_galaxy_manifest_reports_malformed_csv(tmp_path, load_service_module):
    validator = load_service_module(
        "galaxy_manifest_validator_malformed_csv",
        "scripts/validate_galaxy_manifest.py",
    )
    manifest_path = tmp_path / "malformed.csv"
    manifest_path.write_text(
        "image_id,image_path,label,split,source\n"
        '"bad-005,processed/bad-005.jpg,spiral,train,test\n',
        encoding="utf-8",
    )

    errors = validator.validate_manifest(manifest_path)

    assert len(errors) == 1
    assert "Could not read manifest" in errors[0]
    assert "unexpected end of data" in errors[0]
