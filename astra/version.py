"""Fingerprint Python source loaded by a MARTI server process."""

import hashlib
from pathlib import Path


def source_fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


if __name__ == "__main__":
    print(source_fingerprint())
