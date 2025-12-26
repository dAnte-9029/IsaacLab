import sys
from pathlib import Path


def pytest_configure():
    # Allow importing extensions from `source/` without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "source"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

