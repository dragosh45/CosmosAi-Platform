#!/usr/bin/env python3

# Docs: docs/architecture.md Step 1 and docs/excalidraw/manifest_tooling_code_flow.excalidraw explain this validator.

# Import argparse to support validating any manifest path from the command line.
import argparse

# Import sys so the command can return a success or failure exit code.
import sys

# Import Path to handle filesystem paths clearly.
from pathlib import Path


# Add the repo root so this wrapper and training use one validator implementation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-export contract values for old imports while delegating behavior to the package.
from cosmosai.galaxy.labels import ALLOWED_LABELS, ALLOWED_SPLITS  # noqa: E402
from cosmosai.galaxy.manifest import REQUIRED_COLUMNS, validate_manifest  # noqa: E402


# Parse command-line arguments for the validator script.
def parse_args() -> argparse.Namespace:
    # Create the parser with a short explanation for --help output.
    parser = argparse.ArgumentParser(
        description="Validate a CosmosAI galaxy manifest CSV file."
    )

    # Accept a manifest path, defaulting to the tiny sample manifest in this repo.
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/samples/galaxy_manifest_sample.csv",
        help="Path to the galaxy manifest CSV file to validate.",
    )

    # Return the parsed arguments to the caller.
    return parser.parse_args()


# Run the validator as a command-line program.
def main() -> int:
    # Read the manifest path argument from the command line.
    args = parse_args()

    # Validate the manifest and collect all errors.
    errors = validate_manifest(Path(args.manifest_path))

    # Print errors and return a non-zero exit code when validation fails.
    if errors:
        print("Galaxy manifest validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    # Print a short success message when the manifest matches the contract.
    print(f"Galaxy manifest is valid: {args.manifest_path}")
    return 0


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    sys.exit(main())
