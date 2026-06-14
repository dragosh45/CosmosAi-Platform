# Import importlib utilities so tests can load service files with unique module names.
import importlib.util

# Import Path to build stable file paths from the repository root.
from pathlib import Path

# Import pytest so the shared loader can be exposed as a test fixture.
import pytest


# Store the repository root so tests can find app files from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


# Provide a reusable loader to tests that need to import service main.py files.
@pytest.fixture
def load_service_module():
    # Load a service main.py file as a uniquely named Python module for tests.
    def _load_service_module(module_name: str, relative_path: str):
        # Build the absolute path to the service file being tested.
        module_path = REPO_ROOT / relative_path

        # Create an import specification for the service file.
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        # Fail clearly if Python cannot build an import spec for the file.
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path}")

        # Create a module object from the import specification.
        module = importlib.util.module_from_spec(spec)

        # Execute the service file so its FastAPI app, models, and functions are available.
        spec.loader.exec_module(module)

        # Return the loaded module to the test.
        return module

    # Return the inner loader function so each test can choose the service file.
    return _load_service_module
