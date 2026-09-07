#!/usr/bin/env python3
"""Fail if any given file contains machine-specific absolute paths, leaked
personal/internal identities, or hard-coded credentials.

Ported from the USGS ``gspy`` release-hygiene check. Sphinx/sphinx-gallery bakes
the builder's absolute paths into generated docs (tracebacks, warnings), which
then get committed under ``docs/``; the same docs and metadata can also carry
personal free-mail addresses or internal DOI hostnames. Notebooks and example
scripts are the usual route for a stray password or token. This checker blocks
all three classes from entering the repository.

Absolute-path patterns (all operating systems):
  * ``/Users/<name>``            macOS home directories
  * ``/home/<name>/``            Linux home directories
  * ``<drive>:\\Users\\``          Windows profile paths (back- or forward-slash)

Identity-leak patterns:
  * personal free-mail addresses  (gmail/yahoo/hotmail/outlook/aol/icloud/...)
  * internal DOI/USGS hostnames   (``*.gs.doi.net``, ``igsk*`` build machines)

Credential patterns:
  * ``password``/``secret``/``api_key``/``token``/... assigned a quoted literal
  * credentials embedded in a URL  (a ``user:password`` pair before an ``@`` host)
  * PEM private-key blocks, AWS access-key ids, GitHub personal-access tokens

Deliberately NOT flagged, to avoid false positives:
  * System roots such as ``/usr/``, ``/System/``, ``/opt/`` (not user-specific)
  * JSON/string escapes like ``e:\\n`` in notebooks (the ``Users`` segment is
    required after a Windows drive letter)
  * Official ``@usgs.gov`` / ``@contractor.usgs.gov`` / ``@doi.gov`` addresses
    (these are the intended public contact/author identities)
  * Credential values read from the environment or left as placeholders --
    anything holding ``<``, ``>``, ``{``, ``}`` or ``$`` passes, so
    ``password = "<placeholder>"`` and f-string lookups do not trip the check

Usage:
    hygiene.py FILE [FILE ...]

Exit status is non-zero when any offending pattern is found, listing each hit as
``path:line: <match>`` so it doubles as a pre-commit hook and a manual check.
"""
from __future__ import annotations

import re
import sys

# --- Absolute paths -------------------------------------------------------
# One matched group per OS-specific home-directory style. A real name segment
# must follow the separator, so a genuine leak (a real username after the
# separator) is caught while documentation placeholders like ``/Users/<name>``
# are not (which lets this file, and docs describing the rule, pass their own
# check).
_NAME = r"[A-Za-z0-9._-]"
PATH_PATTERN = re.compile(
    rf"/Users/{_NAME}|/home/{_NAME}+/|[A-Za-z]:[\\/]+Users[\\/]+{_NAME}"
)

# --- Identity leaks -------------------------------------------------------
# Personal free-mail addresses that should never be a committed contact/author
# identity in a USGS release. Official ``*.usgs.gov`` / ``doi.gov`` addresses are
# intentionally absent so they pass.
_FREEMAIL_DOMAINS = (
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "me.com", "protonmail.com", "live.com",
)
FREEMAIL_PATTERN = re.compile(
    rf"[A-Za-z0-9._%+-]+@(?:{'|'.join(re.escape(d) for d in _FREEMAIL_DOMAINS)})",
    re.IGNORECASE,
)

# Internal DOI/USGS network hostnames: the ``*.gs.doi.net`` domain and the
# ``igsk...`` build-machine naming convention. These leak internal
# infrastructure and must not ship in a public release.
# A real name segment must precede ``.gs.doi.net`` so documentation
# placeholders like ``*.gs.doi.net`` (and this file's own comments) pass, while
# a genuine ``<hostname>.gs.doi.net`` leak is caught.
HOSTNAME_PATTERN = re.compile(
    r"[A-Za-z0-9._-]+\.gs\.doi\.net|\bigsk[a-z0-9]{4,}",
    re.IGNORECASE,
)

# --- Credentials ----------------------------------------------------------
# A secret-ish key assigned a *quoted literal* of at least six characters. The
# value class excludes ``<>{}$`` so placeholders (``"<your-token>"``), format
# templates and f-string/env lookups pass, and requiring the quotes keeps
# ordinary code such as ``token = tokens[i]`` from tripping.
_SECRET_KEYS = (
    r"pass(?:wo?rd)?|passwd|secret|api[_-]?key|access[_-]?key"
    r"|auth[_-]?token|token|credentials?"
)
SECRET_PATTERN = re.compile(
    rf"(?:{_SECRET_KEYS})\s*[:=]\s*[\"'][^\"'\n<>{{}}$]{{6,}}[\"']",
    re.IGNORECASE,
)

# Credentials baked into a URL, plus the fixed-shape secrets worth matching
# verbatim: PEM private keys, AWS access-key ids, GitHub tokens.
URL_CREDENTIAL_PATTERN = re.compile(
    r"\b[a-z][a-z0-9+.-]*://[^\s:@/]+:[^\s:@/]+@",
    re.IGNORECASE,
)
KNOWN_SECRET_PATTERN = re.compile(
    r"-----BEGIN(?: [A-Z]+)* PRIVATE KEY-----"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
)

# (label, compiled pattern) pairs applied to every line.
CHECKS = (
    ("absolute path", PATH_PATTERN),
    ("personal email", FREEMAIL_PATTERN),
    ("internal hostname", HOSTNAME_PATTERN),
    ("hard-coded credential", SECRET_PATTERN),
    ("credential in URL", URL_CREDENTIAL_PATTERN),
    ("known secret format", KNOWN_SECRET_PATTERN),
)

# Extensions treated as binary and skipped outright. Beyond the web/doc assets,
# this covers the data formats these packages read and write in their tests and
# examples (NetCDF/HDF5, GeoTIFF, shapefile siblings, LAS/LAZ point clouds,
# numpy dumps).
BINARY_EXTENSIONS = (
    ".zip", ".h5", ".hdf5", ".nc", ".nc4", ".netcdf", ".png", ".jpg", ".jpeg",
    ".gif", ".pdf", ".tif", ".tiff", ".ico", ".woff", ".woff2", ".ttf", ".eot",
    ".las", ".laz", ".shp", ".shx", ".dbf", ".npy", ".npz", ".so", ".nbi",
    ".nbc",
)


def file_is_binary(data: bytes) -> bool:
    """Treat a file as binary if it has a NUL byte in its first block."""
    return b"\x00" in data[:4096]


def scan(path: str) -> list[tuple[int, str, str, str]]:
    """Return ``(line_number, label, match, line_text)`` for each offending line."""
    if path.lower().endswith(BINARY_EXTENSIONS):
        return []
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return []
    if file_is_binary(data):
        return []

    text = data.decode("utf-8", "replace")
    hits: list[tuple[int, str, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in CHECKS:
            match = pattern.search(line)
            if match:
                hits.append(
                    (line_number, label, match.group(0), line.strip()[:120])
                )
    return hits


def main(argv: list[str]) -> int:
    found = False
    for path in argv:
        for line_number, label, match, line in scan(path):
            found = True
            print(f"{path}:{line_number}: {label} [{match}] -> {line}")
    if found:
        print(
            "\nRelease-hygiene violation: machine-specific absolute paths, "
            "personal emails, internal hostnames, or hard-coded credentials "
            "detected. Remove them before committing (see scripts/hygiene.py "
            "for the rules).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
