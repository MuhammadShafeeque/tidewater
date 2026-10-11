import re
from pathlib import Path

import tidewater

ROOT = Path(tidewater.__file__).parent
HOSTS = ("oggm", "pygem", "salem")
IMPORT = re.compile(r"^\s*(?:import|from)\s+(%s)\b" % "|".join(HOSTS),
                    re.MULTILINE)
BANNED = ("/home/", "/work/", "/scratch/", "/users/", "\u2014")


def package_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if "__pycache__" in rel.parts or path.suffix in (".pyc", ".nc", ".npz"):
            continue
        yield path, rel


def test_only_its_adapter_imports_a_host():
    """A host is imported in ``hosts/<host>/`` only; the core imports none."""
    hits = []
    for path, rel in package_files():
        if path.suffix != ".py":
            continue
        allowed = rel.parts[1] if rel.parts[0] == "hosts" else None
        hits += [f"{rel}: {m.group(0).strip()}"
                 for m in IMPORT.finditer(path.read_text())
                 if m.group(1) != allowed]
    assert not hits, hits


def test_no_private_tokens():
    hits = []
    for path, rel in package_files():
        try:
            text = path.read_text().lower()
        except UnicodeDecodeError:
            continue
        hits += [f"{rel}: {token!r}" for token in BANNED if token in text]
    assert not hits, hits


def test_import_is_light():
    assert tidewater.__version__
    assert not hasattr(tidewater, "oggm")
