"""Detect declared build dependencies by scanning a project's build
manifests — used to gate framework-specific extractors.

A Scala project isn't necessarily an Akka project: Spark and Delta are
Scala with no ``akka-http``. Rather than run — and announce — Akka HTTP
route extraction on every Scala corpus, gate it on the ``akka-http``
dependency actually appearing in the build config (sbt / maven / gradle).

Scan scope: build manifests at the project root and one level of
sub-modules (the common multi-module sbt/maven/gradle layouts), plus the
sbt ``project/`` dir. A dependency declared only deeper in the tree, or via
an external version catalog, isn't detected — the gate then skips
extraction, which is the intended fail-safe: don't run a framework
extractor without evidence the framework is present.
"""
from __future__ import annotations

from pathlib import Path

# JVM build manifests, at the root and one sub-level (multi-module builds),
# plus the sbt ``project/`` dir where ``Dependencies.scala`` often lives.
_SCALA_BUILD_GLOBS: tuple[str, ...] = (
    'build.sbt', '*/build.sbt',
    'project/*.sbt', 'project/*.scala',
    'pom.xml', '*/pom.xml',
    'build.gradle', 'build.gradle.kts',
    '*/build.gradle', '*/build.gradle.kts',
)


def _manifest_text(root: Path, globs: tuple[str, ...]) -> str:
    """Concatenated text of the matching build manifests under ``root``."""
    parts: list[str] = []
    for pattern in globs:
        for path in root.glob(pattern):
            try:
                parts.append(path.read_text(encoding='utf-8', errors='replace'))
            except OSError:
                continue
    return '\n'.join(parts)


def scala_build_declares(root: Path, token: str) -> bool:
    """True if any JVM build manifest at/under ``root`` (root + one
    sub-level + ``project/``) contains ``token`` — typically an artifact id
    like ``'akka-http'``. Substring match: dependency coordinates carry the
    artifact id verbatim across sbt (``"g" %% "a" % v``), maven
    (``<artifactId>a</artifactId>``), and gradle (``g:a:v``) syntaxes.
    """
    return token in _manifest_text(Path(root), _SCALA_BUILD_GLOBS)


def uses_akka_http(root: Path) -> bool:
    """True if the project declares an ``akka-http`` dependency — the marker
    for running the Akka HTTP route extractor. Matches ``akka-http``,
    ``akka-http-core``, ``akka-http_2.13`` etc. (all carry the substring).
    """
    return scala_build_declares(Path(root), 'akka-http')


__all__ = ['scala_build_declares', 'uses_akka_http']
