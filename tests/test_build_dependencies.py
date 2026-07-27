"""Contract for build-manifest dependency detection — the gate that keeps
Akka HTTP route extraction off Scala projects that don't use akka-http
(Spark, Delta, …)."""
from __future__ import annotations

from pathlib import Path

from docgen.build_dependencies import scala_build_declares, uses_akka_http


def test_sbt_with_akka_http_detected(tmp_path: Path) -> None:
    (tmp_path / 'build.sbt').write_text(
        'libraryDependencies += "com.typesafe.akka" %% "akka-http" % "10.5.0"\n',
    )
    assert uses_akka_http(tmp_path)


def test_plain_scala_project_not_detected(tmp_path: Path) -> None:
    """A Scala project without akka-http (the Spark/Delta case) is skipped."""
    (tmp_path / 'build.sbt').write_text(
        'libraryDependencies += "org.apache.spark" %% "spark-core" % "4.0.0"\n',
    )
    assert not uses_akka_http(tmp_path)


def test_maven_pom_akka_http_detected(tmp_path: Path) -> None:
    (tmp_path / 'pom.xml').write_text(
        '<dependency><groupId>com.typesafe.akka</groupId>'
        '<artifactId>akka-http_2.13</artifactId></dependency>',
    )
    assert uses_akka_http(tmp_path)


def test_submodule_build_file_scanned(tmp_path: Path) -> None:
    (tmp_path / 'build.sbt').write_text('name := "root"\n')
    mod = tmp_path / 'server'
    mod.mkdir()
    (mod / 'build.sbt').write_text(
        '"com.typesafe.akka" %% "akka-http" % akkaHttpV\n',
    )
    assert uses_akka_http(tmp_path)


def test_no_build_files_not_detected(tmp_path: Path) -> None:
    assert not uses_akka_http(tmp_path)


def test_generic_token_match_is_framework_agnostic(tmp_path: Path) -> None:
    (tmp_path / 'build.sbt').write_text('"org.http4s" %% "http4s-core" % v\n')
    assert scala_build_declares(tmp_path, 'http4s')
    assert not scala_build_declares(tmp_path, 'akka-http')
