#!/usr/bin/env python3
"""Canonical docs contract gate for OSS release quality.

Checks:
1) Canonical files exist.
2) Local markdown links resolve.
3) Canonical docs are infographic-first (mermaid + table).
4) API surface docs are fresh vs implemented server endpoints.
5) Core CLI commands are documented and still present in cli/main.cpp usage text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent

CANONICAL_DOCS: Sequence[Path] = (
    ROOT / "README.md",
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "Quickstart.md",
    ROOT / "docs" / "API_SURFACE.md",
    ROOT / "docs" / "UserGuide.md",
    ROOT / "docs" / "AdminGuide.md",
    ROOT / "docs" / "DeveloperGuide.md",
    ROOT / "docs" / "DOCS_STYLE_GUIDE.md",
    ROOT / "docs" / "ARCHIVE_INDEX.md",
)

INFOGRAPHIC_DOCS: Sequence[Path] = (
    ROOT / "docs" / "INDEX.md",
    ROOT / "docs" / "Quickstart.md",
    ROOT / "docs" / "API_SURFACE.md",
    ROOT / "docs" / "UserGuide.md",
    ROOT / "docs" / "AdminGuide.md",
    ROOT / "docs" / "DeveloperGuide.md",
    ROOT / "docs" / "ARCHIVE_INDEX.md",
)

API_DOC = ROOT / "docs" / "API_SURFACE.md"
SERVER_HTTP_CPP = ROOT / "server" / "http" / "http_server.cpp"
CLI_MAIN_CPP = ROOT / "cli" / "main.cpp"

CORE_CLI_MARKERS: Sequence[str] = (
    "inferctl completion",
    "inferctl chat",
    "inferctl models",
    "inferctl admin models",
    "inferctl admin routing",
    "inferctl admin pools",
)

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
PATH_METHOD_SINGLE_RE = re.compile(
    r'if\s*\(\s*method\s*==\s*"([A-Z]+)"\s*&&\s*path\s*==\s*"([^"]+)"\s*\)'
)
PATH_METHOD_OR_RE = re.compile(
    r'if\s*\(\s*method\s*==\s*"([A-Z]+)"\s*&&\s*\(\s*path\s*==\s*"([^"]+)"\s*\|\|\s*path\s*==\s*"([^"]+)"\s*\)\s*\)',
    re.DOTALL,
)


def fail(msg: str) -> None:
  print(f"[docs-gate] FAIL: {msg}")
  raise SystemExit(1)


def read_text(path: Path) -> str:
  try:
    return path.read_text(encoding="utf-8")
  except FileNotFoundError:
    fail(f"missing file: {path.relative_to(ROOT)}")
  except UnicodeDecodeError as exc:
    fail(f"utf-8 decode error in {path.relative_to(ROOT)}: {exc}")
  return ""  # unreachable


def assert_files_exist(paths: Iterable[Path]) -> None:
  missing = [p for p in paths if not p.exists()]
  if missing:
    fail(
        "missing canonical docs: "
        + ", ".join(str(p.relative_to(ROOT)) for p in missing)
    )


def assert_links_resolve(path: Path, text: str) -> None:
  for raw_link in LINK_RE.findall(text):
    if raw_link.startswith(("http://", "https://", "#", "mailto:")):
      continue
    target_raw = raw_link.split("#", 1)[0]
    if not target_raw:
      continue
    target = (path.parent / target_raw).resolve()
    if not target.exists():
      fail(
          f"broken link in {path.relative_to(ROOT)}: {raw_link} "
          f"(resolved to {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target})"
      )


def assert_infographic_contract(path: Path, text: str) -> None:
  if "```mermaid" not in text:
    fail(f"{path.relative_to(ROOT)} must contain at least one mermaid diagram")
  if "|" not in text:
    fail(f"{path.relative_to(ROOT)} must contain at least one markdown table")


def extract_http_endpoints(server_text: str) -> Set[Tuple[str, str]]:
  pairs: Set[Tuple[str, str]] = set()
  for method, path in PATH_METHOD_SINGLE_RE.findall(server_text):
    pairs.add((method, path))
  for method, path_a, path_b in PATH_METHOD_OR_RE.findall(server_text):
    pairs.add((method, path_a))
    pairs.add((method, path_b))

  # /v1/models is handled with prefix matching logic.
  if 'path == "/v1/models"' in server_text:
    pairs.add(("GET", "/v1/models"))
  if '"/v1/models/"' in server_text:
    pairs.add(("GET", "/v1/models/{id}"))

  allowed_prefixes = ("/v1/", "/livez", "/readyz", "/healthz", "/metrics", "/ui")
  return {p for p in pairs if p[1].startswith(allowed_prefixes)}


def assert_api_docs_fresh(server_pairs: Set[Tuple[str, str]], api_doc_text: str) -> None:
  lines = api_doc_text.splitlines()
  for method, path in sorted(server_pairs):
    candidate_lines = [line for line in lines if path in line]
    if not candidate_lines:
      fail(f"docs/API_SURFACE.md missing endpoint path: {path} ({method})")
    if not any(method in line for line in candidate_lines):
      fail(
          f"docs/API_SURFACE.md missing method '{method}' for endpoint '{path}'"
      )


def assert_cli_docs_fresh(cli_text: str, docs_text_joined: str) -> None:
  for marker in CORE_CLI_MARKERS:
    if marker not in cli_text:
      fail(f"cli/main.cpp no longer contains usage marker: '{marker}'")
    if marker not in docs_text_joined:
      fail(f"canonical docs missing CLI marker: '{marker}'")


def main() -> int:
  print("[docs-gate] checking canonical docs contract...")

  assert_files_exist(CANONICAL_DOCS)

  docs_texts: List[str] = []
  for path in CANONICAL_DOCS:
    text = read_text(path)
    docs_texts.append(text)
    assert_links_resolve(path, text)

  for path in INFOGRAPHIC_DOCS:
    text = read_text(path)
    assert_infographic_contract(path, text)

  server_text = read_text(SERVER_HTTP_CPP)
  api_doc_text = read_text(API_DOC)
  server_pairs = extract_http_endpoints(server_text)
  if not server_pairs:
    fail("failed to extract HTTP endpoint contracts from server/http/http_server.cpp")
  assert_api_docs_fresh(server_pairs, api_doc_text)

  cli_text = read_text(CLI_MAIN_CPP)
  joined_docs = "\n".join(docs_texts)
  assert_cli_docs_fresh(cli_text, joined_docs)

  print(f"[docs-gate] canonical docs checked: {len(CANONICAL_DOCS)}")
  print(f"[docs-gate] endpoint contracts checked: {len(server_pairs)}")
  print(f"[docs-gate] cli markers checked: {len(CORE_CLI_MARKERS)}")
  print("[docs-gate] PASSED")
  return 0


if __name__ == "__main__":
  sys.exit(main())
