from __future__ import annotations

import py_compile
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather",
    ".joblib", ".pkl", ".pickle",
    ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff",
    ".doc", ".docx", ".ppt", ".pptx",
}
ALLOWED_TOP_LEVEL = {
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "config",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "scripts",
    "tests",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".cff", ""}


def repository_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        path = ROOT / name
        if path.is_file():
            yield path


def test_release_file_types_and_top_level():
    assert {p.relative_to(ROOT).parts[0] for p in repository_files()} <= ALLOWED_TOP_LEVEL
    forbidden = [str(p.relative_to(ROOT)) for p in repository_files()
                 if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not forbidden, f"Forbidden release artefacts: {forbidden}"


def test_python_sources_compile(tmp_path):
    for script in sorted((ROOT / "scripts").glob("*.py")):
        py_compile.compile(
            str(script),
            cfile=str(tmp_path / f"{script.stem}.pyc"),
            doraise=True,
        )


def test_no_private_absolute_paths():
    hits = []
    for path in repository_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("/public" + "/home/", "v5" + "_report", "revision_" + "ti_strict"):
            if marker in text:
                hits.append(f"{path.relative_to(ROOT)}: {marker}")
    assert not hits, f"Private path markers found: {hits}"


def test_no_common_credentials():
    patterns = [
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"ghp_[A-Za-z0-9]{20,}"),
        re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"(?i)(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]+['\"]"),
    ]
    hits = []
    for path in repository_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                hits.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    assert not hits, f"Potential credentials found: {hits}"


def test_required_repository_metadata():
    for name in ("README.md", "LICENSE", "CITATION.cff", "requirements.txt"):
        assert (ROOT / name).is_file()
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
