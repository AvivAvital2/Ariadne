"""Spool pack container — build and install.

A pack is ONE zip: ``manifest.yaml`` (whose ``checksum`` covers the DB
inside) + ``pack.db`` — a genuine Ariadne library DB holding exactly the
spool source's documents, embeddings included. Matching embedding spaces
reuse vectors; other spaces re-embed every embedded record before import.
"""
import hashlib
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import Context
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from config import OFFICIAL_DOC_PROVENANCE
from docgen.extraction_coverage import EXTRACTION_COVERAGE_VERSION
from spools import PACK_INSTALL_PENDING, SpoolError, SpoolManifest, spool_source_id

if TYPE_CHECKING:
    from library import Library
    from schema import Document

# The pack container format this Ariadne writes and reads. Bump on any
# incompatible pack.db/manifest layout change; install fails loud on a
# mismatch instead of leaning on implicit schema migrations.
PACK_FORMAT = 1

# Bounded fetch batch for build (HIGH-3 — never load a whole source at once).
_PACK_BATCH = 500

# CRIT-7 input bounds for install. A pack is (designed to be) fetched from a
# remote location, so its zip is untrusted: read nothing unbounded. pack.db is
# STREAMED to disk in chunks (memory stays O(chunk) regardless of size); the
# byte cap then guards against a decompression-bomb filling the disk. Manifest
# and member count are tiny by nature — capped so a lying/huge header or an
# anchor-bomb YAML can't blow up before verification.
_INSTALL_CHUNK = 1024 * 1024                 # 1 MB streaming chunk
_DEFAULT_MAX_PACK_BYTES = 32 * 1024 ** 3     # 32 GB — generous vs a real corpus
_DEFAULT_MAX_MANIFEST_BYTES = 1024 * 1024    # 1 MB — manifests are tiny
_MAX_ZIP_MEMBERS = 256                       # manifest + pack.db + license members
_MAX_LICENSE_BYTES = 1024 * 1024             # 1 MB — license/notice files are tiny
# Top-level filenames captured for upstream attribution (§18.1).
_ATTRIBUTION_FILENAMES = ('license', 'licence', 'copying', 'notice')


def _sha256(blob: bytes) -> str:
    return 'sha256:' + hashlib.sha256(blob).hexdigest()


def _checkpoint(db_path: Path) -> None:
    """Fold the WAL back into the main DB file so its bytes are complete."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        conn.close()


def _under_certified(source_file: str, source_root: Path, certified_docs) -> bool:
    """True iff ``source_file`` falls under one of the certified doc dirs
    (paths relative to the environment repo root — §18.6.2 tagging rule)."""
    try:
        rel = Path(source_file).resolve().relative_to(source_root)
    except ValueError:
        return False
    rel_posix = rel.as_posix()
    for cert in certified_docs:
        prefix = cert.strip('/')
        if rel_posix == prefix or rel_posix.startswith(prefix + '/'):
            return True
    return False


def _strip_builder_paths(doc, root_prefix: str):
    """Builder-machine absolute paths must not ship in a pack: rewrite
    ``source_root``-absolute references to corpus-RELATIVE in ``source_files``
    and in the content text (catalog bodies embed the path string). Shipping
    them leaks the builder's username/machine layout — a real pack carried
    them in 100% of its docs — and a consumer's file reads that follow them
    dead-end anyway (the corpus isn't on the consumer's disk)."""
    from attrs import evolve
    return evolve(
        doc,
        content=doc.content.replace(root_prefix, ''),
        source_files=[
            f[len(root_prefix):] if f.startswith(root_prefix) else f
            for f in doc.source_files
        ],
    )


def _artifact_library(path: Path) -> 'Library':
    """A pack is a data container, independent of the active consumer database."""
    from library import Library
    return Context().run(Library, path)


def _embedding_dimension(library: 'Library') -> int | None:
    """Inspect all carried vectors, including children of unembedded documents."""
    with library._conn_provider.acquire() as connection:
        widths = {row[0] for table in ('documents', 'chunks', 'sections')
                  for row in connection.execute(
                      f'SELECT DISTINCT length(embedding) FROM {table} WHERE embedding IS NOT NULL')}
    if len(widths) > 1 or any(width <= 0 or width % 4 for width in widths):
        raise SpoolError('Pack contains inconsistent embedding dimensions')
    return next(iter(widths)) // 4 if widths else None


def _needs_reembedding(manifest: SpoolManifest, consumer_identity: dict | None, dimension: int | None) -> bool:
    import embedding

    if (manifest.embedding_dim is not None and manifest.embedding_dim != dimension
            or manifest.embedding_identity is not None and dimension is not None
            and manifest.embedding_identity['dimensions'] != dimension):
        raise SpoolError('Pack embedding dimensions disagree with its manifest')
    if dimension is None:
        return False
    if manifest.embedding_identity is not None:
        return manifest.embedding_identity != consumer_identity
    if consumer_identity is not None:
        return (consumer_identity['provider'] == 'local'
                or dimension != consumer_identity['dimensions']
                or manifest.embedding_model != consumer_identity['model'])
    return (dimension != embedding.EMBEDDING_DIM
            or bool(manifest.embedding_model and manifest.embedding_model != embedding.DEFAULT_MODEL))


def _check_install_ownership(library: 'Library', pack: 'Library', docs: list['Document'], source_name: str) -> set[str]:
    association = spool_source_id(source_name)
    incoming = pack.list_themes(coherent_only=False, association=association)
    theme_ids = {theme.doc_id for theme in incoming}
    owned = library.list_themes(coherent_only=False, association=association)
    owned_ids = {theme.doc_id for theme in owned}
    owned_clusters = {theme.cluster_id for theme in owned}
    for theme in incoming:
        if library.get_theme(theme.cluster_id) is not None and theme.cluster_id not in owned_clusters:
            raise SpoolError(f'Theme {theme.cluster_id} belongs to another source')
    for doc in docs:
        expected = None if doc.id in theme_ids else source_name
        if doc.source_name != expected:
            raise SpoolError(f'Pack document {doc.id} does not belong to its declared source')
        previous = library.get_document(doc.id)
        if previous is not None and (previous.source_name != association
                                     if doc.id not in theme_ids else doc.id not in owned_ids):
            raise SpoolError(f'Document {doc.id} belongs to another source')
    return theme_ids


def _copy_doc_with_sections(dest, doc, *, embedding, metadata, source_name,
                            sections, chunks=(), allow_reserved_source=False, artifact=False,
                            connection=None):
    """Copy one document (+ its sections) into ``dest`` — the shared body of
    the build and install copy loops, differing only in these values.

    ``allow_reserved_source`` is the install side's privilege to write under the
    reserved ``spool:`` namespace (post-verification); the build side leaves it
    False (it writes the plain environment source name)."""
    from spools import is_spool_source
    if is_spool_source(source_name) and not allow_reserved_source:
        raise SpoolError('Reserved spool sources require a verified install')
    context = getattr(dest, 'inference_context', None)
    if context is not None and context.local:
        from attrs import evolve

        from library.prepared import PreparedDocument, apply_document
        previous = dest.get_document(doc.id)
        prepared = PreparedDocument(
            document=evolve(doc, embedding=embedding, metadata=metadata, source_name=source_name),
            chunks=tuple(chunks), sections=tuple(sections or ()),
            expected_version=previous.updated_at if previous else None,
            embedding_space_id=context.settings.embeddings.embedding_space_id)
        if connection is None:
            dest.apply_prepared_document(prepared)
        else:
            context.require_writer()
            context.validate()
            apply_document(connection, prepared)
        return
    if artifact:
        from attrs import evolve
        copied = evolve(doc, embedding=embedding, metadata=metadata, source_name=source_name)
        # Artifact copies are wholly offline: add_document would token-count
        # via the legacy hosted provider and discard the stored timestamps.
        with dest._conn_provider.acquire() as conn:
            conn.execute(
                'INSERT INTO documents '
                '(id,content_type,title,content,source_files,embedding,created_at,updated_at,metadata,source_name) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (copied.id, copied.content_type, copied.title, copied.content,
                 copied.source_files_json(), copied.embedding_bytes(), copied.created_at,
                 copied.updated_at, copied.metadata_json(), copied.source_name))
        dest.add_chunks_batch(list(chunks))
        dest.store_sections(doc.id, list(sections or ()))
        return
    dest.add_document(
        doc.content_type,
        doc.title,
        doc.content,
        source_files=list(doc.source_files),
        embedding=embedding,
        metadata=metadata,
        doc_id=doc.id,
        source_name=source_name,
        _allow_reserved_source=allow_reserved_source,
    )
    if sections:
        dest.store_sections(doc.id, sections)
    if chunks:
        dest.delete_chunks(doc.id)
        dest.add_chunks_batch(list(chunks))


_SCIP_SOURCE_SCOPED_LAYERS = (
    'api_endpoints', 'api_calls', 'config_values', 'string_literals',
    'process_invocations', 'http_client_calls', 'config_reads',
)


def _copy_source_scoped_table(sconn, dconn, table, source_name, target_source,
                              *, strip_path_prefix=None):
    """Copy one ``source_name``-scoped SCIP layer table,
    rewriting the source_name column to ``target_source``.

    Columns are read from the schema (PRAGMA) so every layer copies uniformly;
    the autoincrement ``id`` rowid is dropped (regenerated on insert) and dedup
    rides the table's own UNIQUE constraint via ``INSERT OR IGNORE``. Any
    symbol-id references (canonical_ids) are kept raw, consistent with
    scip_symbols. Table names are a fixed internal allow-list, not input.
    """
    cols = [r[1] for r in sconn.execute(f'PRAGMA table_info({table})')
            if r[1] != 'id']
    if 'source_name' not in cols:
        return
    collist = ', '.join(cols)
    rows = sconn.execute(
        f'SELECT {collist} FROM {table} WHERE source_name = ?', (source_name,),
    ).fetchall()
    if not rows:
        return
    if strip_path_prefix and 'file' in cols:
        # Same no-builder-paths rule as the documents (_strip_builder_paths):
        # string_literals/config_values carried machine-absolute paths in
        # every row of a real pack.
        fi = cols.index('file')
        rows = [
            (*r[:fi],
             (r[fi][len(strip_path_prefix):]
              if isinstance(r[fi], str) and r[fi].startswith(strip_path_prefix)
              else r[fi]),
             *r[fi + 1:])
            for r in rows
        ]
    sn = cols.index('source_name')
    out = [(*r[:sn], target_source, *r[sn + 1:]) for r in rows]
    dconn.executemany(
        f'INSERT OR IGNORE INTO {table} ({collist}) '
        f'VALUES ({", ".join("?" * len(cols))})',
        out,
    )


def _copy_source_scip(src, dest, source_name, *, dest_source_name=None,
                      strip_path_prefix=None):
    """Copy ``source_name``'s self-contained SCIP graph from ``src`` into
    ``dest``: its ``scip_symbols`` (scoped by source_name)
    and every ``scip_edges`` row touching one of those symbols.

    ``dest_source_name`` (install side) rewrites the ``source_name`` COLUMN to
    the reserved spool id; the edges carry no source column, so they copy as
    is. canonical_ids are copied VERBATIM — the spool's own edges reference
    them, so the shipped graph is self-consistent, and there is no weld to the
    consumer's graph to preserve. ``INSERT OR IGNORE`` never
    clobbers a colliding row (the global canonical_id PK) — harmless into a
    fresh pack, load-bearing at install.
    """
    target_source = dest_source_name or source_name
    with src._conn_provider.acquire() as sconn, \
            dest._conn_provider.acquire() as dconn:
        sym_rows = sconn.execute(
            'SELECT canonical_id, source_name, language, file, line_start, '
            'line_end, kind, display_name, qualified_name, '
            'parent_qualified_name FROM scip_symbols WHERE source_name = ?',
            (source_name,),
        ).fetchall()
        if sym_rows:
            dconn.executemany(
                'INSERT OR IGNORE INTO scip_symbols (canonical_id, '
                'source_name, language, file, line_start, line_end, kind, '
                'display_name, qualified_name, parent_qualified_name) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                [(r[0], target_source, *r[2:]) for r in sym_rows],
            )
            # Chunk the id list so a large corpus's symbol set can't exceed
            # SQLite's bind-variable limit (each chunk is bound twice by the
            # caller/callee OR → copies=2). An edge whose caller and callee land
            # in different chunks is fetched twice; the INSERT OR IGNORE below
            # dedupes it on the edge primary key.
            from library.sql_vars import chunk_ids
            ids = [r[0] for r in sym_rows]
            edge_rows = []
            for chunk in chunk_ids(ids, copies=2):
                placeholders = ','.join('?' * len(chunk))
                edge_rows.extend(sconn.execute(
                    'SELECT caller_canonical_id, callee_canonical_id, edge_type, '
                    f'file, line, confidence FROM scip_edges WHERE '
                    f'caller_canonical_id IN ({placeholders}) OR '
                    f'callee_canonical_id IN ({placeholders})',
                    chunk + chunk,
                ).fetchall())
            dconn.executemany(
                'INSERT OR IGNORE INTO scip_edges (caller_canonical_id, '
                'callee_canonical_id, edge_type, file, line, confidence) '
                'VALUES (?,?,?,?,?,?)',
                edge_rows,
            )
        from docgen.scip_store import backfill_canonical_ownership
        backfill_canonical_ownership(dconn, target_source)
        # The remaining source-scoped SCIP layers, copied uniformly
        # (runs even when the corpus has no symbols).
        for table in _SCIP_SOURCE_SCOPED_LAYERS:
            _copy_source_scoped_table(
                sconn, dconn, table, source_name, target_source,
                strip_path_prefix=strip_path_prefix,
            )


def _copy_version_facts(src, dest, source_name):
    """Copy the corpus's version_facts (Slice 2). Facts are keyed by the
    CORPUS source name end-to-end (the entity indexes key symbols the same
    way), so install copies without renaming."""
    from library.version_facts import init_version_facts_schema
    with src._conn_provider.acquire() as sconn, \
            dest._conn_provider.acquire() as dconn:
        init_version_facts_schema(sconn)
        init_version_facts_schema(dconn)
        _copy_source_scoped_table(
            sconn, dconn, 'version_facts', source_name, source_name)


def _copy_spool_themes(src, dest, spool_name, *, artifact=False, copy_documents=True) -> list:
    """Ship the spool's OWN theme pass (docs + themes rows + members —
    Slice 3). Theme docs are cross-source by design (source_name=NULL) and
    gate by ASSOCIATION: it is already the reserved spool id
    (build_spool_internal_themes), so the consumer's scope rule admits them
    iff the spool is enabled. Base-pass themes never ship. Existing rows
    for the association are replaced (idempotent reinstall). Returns the
    doc ids newly inserted into ``dest`` (the install rollback tracks them).
    """
    from spools import spool_source_id

    association = spool_source_id(spool_name)
    themes = src.list_themes(coherent_only=False, association=association)
    if copy_documents:
        dest.delete_themes_for_association(association)
    else:
        incoming = {theme.doc_id for theme in themes}
        for old in dest.list_themes(coherent_only=False, association=association):
            if old.doc_id not in incoming:
                dest.delete_document(old.doc_id)
        with dest._conn_provider.acquire() as conn:
            conn.execute('DELETE FROM themes WHERE association=?', (association,))
            conn.execute('DELETE FROM spool_assoc_sync WHERE association=?', (association,))
    inserted = []
    for theme in themes:
        doc = src.get_document(theme.doc_id)
        if doc is None:
            continue
        if dest.get_document(doc.id) is None:
            inserted.append(doc.id)
        if copy_documents:
            _copy_doc_with_sections(
                dest, doc, embedding=doc.embedding, metadata=dict(doc.metadata),
                source_name=None, sections=src.get_sections(doc.id),
                chunks=src.get_chunks(doc.id), artifact=artifact,
            )
        dest.add_theme(
            cluster_id=theme.cluster_id, doc_id=theme.doc_id,
            member_count=theme.member_count, resolution=theme.resolution,
            summary_hash=theme.summary_hash, coherent=theme.coherent,
            dirty=theme.dirty, association=association,
        )
        dest.set_theme_members(
            theme.cluster_id, src.get_theme_members(theme.cluster_id))
    return inserted


def _stage_records(pack: 'Library', docs: list['Document'], library: 'Library', *,
                   reembed: bool, pack_path: Path) -> dict:
    """Prepare every imported vector before modifying the consumer library."""
    from attrs import evolve

    from embedding import doc_embedding_text, embed_batch_sync

    records = {doc.id: [doc, pack.get_chunks(doc.id), pack.get_sections(doc.id)] for doc in docs}
    pending = []
    texts = []
    if reembed:
        for doc, chunks, sections in records.values():
            if doc.embedding is not None:
                pending.append((doc.id, 0, None, doc))
                texts.append(doc_embedding_text(doc.title, doc.content))
            for kind, children in ((1, chunks), (2, sections)):
                for index, child in enumerate(children):
                    if child.embedding is not None:
                        pending.append((doc.id, kind, index, child))
                        texts.append(child.content if kind == 1 else child.heading or 'Introduction')
        if texts:
            context = getattr(library, 'inference_context', None)
            try:
                with context.activate() if context else nullcontext():
                    vectors = embed_batch_sync(texts, **({'purpose': 'document'} if context else {}))
            except Exception as exc:
                raise SpoolError(
                    f'pack {pack_path}: re-embedding under the consumer model failed ({exc}); '
                    'ensure the embedding API is configured') from exc
            if len(vectors) != len(pending):
                raise SpoolError(f'pack {pack_path}: re-embed returned {len(vectors)} vectors '
                                 f'for {len(pending)} records — refusing')
            for (doc_id, kind, index, record), vector in zip(pending, vectors, strict=True):
                updated = evolve(record, embedding=vector)
                if index is None:
                    records[doc_id][kind] = updated
                else:
                    records[doc_id][kind][index] = updated
    context = getattr(library, 'inference_context', None)
    if context is not None and context.local:
        from library.prepared import validate_vector
        for doc, chunks, sections in records.values():
            for record in (doc, *chunks, *sections):
                validate_vector(record.embedding)
    return records


def _copy_surface_tags(src, dest, source_name):
    """Copy the corpus's surface_tags (Slice 3) — corpus-keyed like
    version_facts; no rename at install."""
    from library.surface_tags import init_surface_tags_schema
    with src._conn_provider.acquire() as sconn, \
            dest._conn_provider.acquire() as dconn:
        init_surface_tags_schema(sconn)
        init_surface_tags_schema(dconn)
        _copy_source_scoped_table(
            sconn, dconn, 'surface_tags', source_name, source_name)


def _license_name(text) -> str | None:
    """The license IDENTIFIER for a license text, via the same marker table
    the license gate uses (never a guess — unknown texts carry no name)."""
    from spools import classify_license
    _category, name = classify_license(text)
    return name


def _gather_attribution(source_root):
    """Scan corpus clones under ``source_root`` (each marked with the fetch's
    ``.ariadne-corpus-sha``) for their top-level LICENSE/NOTICE files, so the
    pack ships upstream attribution (§18.1). Returns ``(records, blobs)``:
    ``records`` is the manifest form — a tuple of
    ``{repo, sha, files: ({name, sha256}, ...)}`` — and ``blobs`` maps each
    ``licenses/<repo>/<file>`` arcname to its bytes for the zip. No corpus
    clones (or none carrying a license) → empty, so a hand-built pack simply
    carries no attribution."""
    from spool_acquire import _CORPUS_SHA_MARKER
    source_root = Path(source_root)
    records = []
    blobs = {}
    for marker in sorted(source_root.glob(f'*/{_CORPUS_SHA_MARKER}')):
        repo_dir = marker.parent
        repo = repo_dir.name
        sha = marker.read_text(encoding='utf-8').strip()
        files = []
        license_name = None
        for entry in sorted(repo_dir.iterdir()):
            low = entry.name.lower()
            if not entry.is_file() or not any(
                    low == n or low.startswith(n + '.') or low.startswith(n + '-')
                    for n in _ATTRIBUTION_FILENAMES):
                continue
            data = entry.read_bytes()
            blobs[f'licenses/{repo}/{entry.name}'] = data
            files.append({'name': entry.name, 'sha256': _sha256(data)})
            if license_name is None and low.startswith('licen'):
                license_name = _license_name(
                    data.decode('utf-8', errors='ignore'))
        if files:
            record = {'repo': repo, 'sha': sha, 'files': tuple(files)}
            if license_name:
                record['license_name'] = license_name
            records.append(record)
    return tuple(records), blobs


def _stage_attribution_licenses(pack_path, attribution):
    """Read + integrity-check each license member declared in ``attribution``,
    returning ``{(repo, name): bytes}``. Each is verified against its recorded
    sha256 BEFORE the caller writes anything to the store/cache (fail-closed).
    Only members DECLARED in the manifest are read (the zip's own member list is
    untrusted); a missing or mismatched member is refused loudly.

    As with the pack.db checksum (§19.2) this proves INTEGRITY, not
    AUTHENTICITY: an attacker who rewrites pack.db can also rewrite a license
    and its recorded sha. Acceptable only for the v1 local-install path — a
    remote fetch needs signature verification first."""
    staged = {}
    if not attribution:
        return staged
    with zipfile.ZipFile(pack_path) as zf:
        present = set(zf.namelist())
        for record in attribution:
            repo = record['repo']
            for entry in record.get('files', ()):
                name = entry['name']
                arc = f'licenses/{repo}/{name}'
                if arc not in present:
                    raise SpoolError(
                        f'pack {pack_path}: attribution declares license {arc} '
                        f'but the member is missing — refusing',
                    )
                blob = _read_capped(zf, arc, _MAX_LICENSE_BYTES)
                digest = _sha256(blob)
                if digest != entry['sha256']:
                    raise SpoolError(
                        f'pack {pack_path}: license {arc} failed integrity '
                        f'check (declared {entry["sha256"]}, got {digest}) — '
                        f'refusing',
                    )
                staged[(repo, name)] = blob
    return staged


@contextmanager
def _staged_cache(cache_dir: str | Path, environment: str, manifest_bytes: bytes,
                  staged_db: Path, licenses: dict[tuple[str, str], bytes]) -> Iterator[Callable[[], None]]:
    """Publish as incomplete, then enable only after the database commit.

    Ordinary exceptions restore the old cache. A process exit leaves the new
    cache disabled; reinstalling safely completes or repeats its library write.
    """
    cache = Path(cache_dir)
    created = not cache.exists()
    cache.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.install-', dir=cache) as temporary:
            workspace = Path(temporary)
            prepared, backup, destination = workspace / 'new', workspace / 'old', cache / environment
            prepared.mkdir()
            (prepared / PACK_INSTALL_PENDING).touch()
            (prepared / 'manifest.yaml').write_bytes(manifest_bytes)
            shutil.copyfile(staged_db, prepared / 'pack.db')
            for (repo, name), blob in licenses.items():
                directory = prepared / 'licenses' / repo
                directory.mkdir(parents=True, exist_ok=True)
                (directory / name).write_bytes(blob)
            published = False

            def publish() -> None:
                nonlocal published
                if destination.exists():
                    destination.replace(backup)
                prepared.replace(destination)
                published = True

            try:
                yield publish
            except BaseException:
                if published:
                    shutil.rmtree(destination)
                if backup.exists():
                    backup.replace(destination)
                raise
            else:
                # The caller's SQLite context has committed before returning
                # here. Failure to remove the marker must leave this cache
                # disabled, rather than restore a cache for older content.
                if published:
                    (destination / PACK_INSTALL_PENDING).unlink()
    finally:
        if created and cache.exists() and not any(cache.iterdir()):
            cache.rmdir()


def build_pack(
    library,
    *,
    environment: str,
    version: str,
    target_runtime: str,
    certified_docs=(),
    source_root,
    out_path,
    corpus_shas=None,
    taxonomy=(),
    runtime_components=None,
    surfaces=None,
    name_aliases=None,
) -> SpoolManifest:
    """Build a pack zip from ``library``'s ``environment`` source.

    The environment id IS the source name — docs are selected by
    ``source_name == environment``, so the installed pack lands under the
    same name ``resolve_spools`` registers (scope-union consistency).
    Certified-dir docs are tagged ``official`` in the PACK only; the
    building store is never mutated.
    """
    # HIGH-3: read ONLY the target source's docs — never
    # ``list_documents()`` (which materializes every source's content +
    # embeddings into memory; fatal on a real multi-GB store). Lite metas
    # are light; full docs are fetched in bounded batches for our ids only.
    metas = list(library.list_documents_lite())
    read_source = environment
    ids = [meta.id for meta in metas if meta.source_name == environment]
    if not ids:
        # Rebuild-from-installed: after install the docs live ONLY under the
        # reserved 'spool:<env>' id (CRIT-9) — read that copy, and write the
        # pack under the PLAIN name below, so install/rebuild stay symmetric
        # inverses and a rebuilt pack is canonical.
        read_source = spool_source_id(environment)
        ids = [meta.id for meta in metas if meta.source_name == read_source]
    if not ids:
        raise SpoolError(
            f'no documents for source {environment!r} (or installed '
            f'{spool_source_id(environment)!r}) — nothing to pack',
        )
    source_root = Path(source_root).resolve()
    # Certification is judged on the ORIGINAL absolute paths below; every
    # written row is then rewritten corpus-relative (_strip_builder_paths).
    root_prefix = str(source_root) + '/'

    # Slice 2: extract the corpus's version markers (from the LOCATED source
    # lines — stored signatures are truncated at the annotation) into
    # version_facts on the builder store; the pack ships the FACTS so a
    # consumer never needs the corpus files. Idempotent.
    from library.version_facts import extract_facts_for_source
    extract_facts_for_source(
        library, outer_sources=[read_source], corpus_source=environment,
        source_root=source_root)
    if surfaces:
        # Slice 3: tag the corpus's doc-grade docs from the recipe's
        # surface vocabularies (deterministic, offline). Idempotent.
        from library.surface_tags import tag_surfaces_for_source
        tag_surfaces_for_source(
            library, outer_sources=[read_source], corpus_source=environment,
            surfaces=surfaces)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / 'pack.db'
        with _artifact_library(db_path) as pack:
            for batch_start in range(0, len(ids), _PACK_BATCH):
                batch = ids[batch_start:batch_start + _PACK_BATCH]
                docs = library.get_documents_batch(batch)
                sections = library.get_sections_batch(batch)
                for doc in docs:
                    metadata = dict(doc.metadata)
                    if any(
                        _under_certified(f, source_root, certified_docs)
                        for f in doc.source_files
                    ):
                        metadata['provenance'] = OFFICIAL_DOC_PROVENANCE
                    shipped = _strip_builder_paths(doc, root_prefix)
                    _copy_doc_with_sections(
                        pack, shipped, embedding=doc.embedding,
                        metadata=metadata,
                        source_name=environment,
                        sections=sections.get(doc.id),  # HIGH-2: sections travel
                        chunks=library.get_chunks(doc.id), artifact=True,
                    )
            _copy_source_scip(library, pack, environment,
                              strip_path_prefix=root_prefix)
            _copy_version_facts(library, pack, environment)
            _copy_surface_tags(library, pack, environment)
            _copy_spool_themes(library, pack, environment, artifact=True)
            embedding_dim = _embedding_dimension(pack)
        _checkpoint(db_path)
        db_blob = db_path.read_bytes()

    # CRIT-10: stamp the embedding identity so install can verify it. The
    # Dimension comes from every embedded record; bound identity includes
    # artifact revision, representation and the full input/vector recipe.
    import embedding as _embedding
    from library.model_binding import inspect_binding
    binding = inspect_binding(library.path)
    embedding_identity = binding.manifest['embedding'] if binding is not None else None
    embedding_model = ((embedding_identity['model'] if embedding_identity else _embedding.DEFAULT_MODEL)
                       if embedding_dim is not None else None)

    attribution, license_blobs = _gather_attribution(source_root)

    manifest = SpoolManifest(
        environment=environment,
        version=str(version),
        target_runtime=target_runtime,
        certified_docs=tuple(certified_docs),
        checksum=_sha256(db_blob),
        pack_format=PACK_FORMAT,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_identity=embedding_identity,
        corpus_shas=dict(corpus_shas or {}),
        taxonomy=tuple(taxonomy or ()),
        runtime_components=dict(runtime_components or {}),
        surfaces={k: list(v) for k, v in (surfaces or {}).items()},
        name_aliases=[str(a) for a in (name_aliases or [])],
        extraction_coverage_version=EXTRACTION_COVERAGE_VERSION,
        attribution=attribution,
    )
    manifest_yaml = yaml.safe_dump(manifest.to_dict(), sort_keys=False)
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('manifest.yaml', manifest_yaml)
        zf.writestr('pack.db', db_blob)
        for arcname, blob in license_blobs.items():
            zf.writestr(arcname, blob)
    return manifest


def _read_capped(zf, name, cap: int) -> bytes:
    """Read a zip member with a hard byte cap, not trusting its header."""
    with zf.open(name) as fh:
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise SpoolError(
            f"pack member {name!r} exceeds the {cap} byte cap — refusing",
        )
    return data


def install_pack(library, pack_path, *, cache_dir,
                 max_pack_bytes: int = _DEFAULT_MAX_PACK_BYTES,
                 max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES,
                 ) -> SpoolManifest:
    """Verify and install a pack zip into ``library`` + the cache.

    The checksum is verified BEFORE anything is written — a tampered pack
    leaves both the store and the cache untouched (fail-closed, §5).
    Matching embeddings travel as stored blobs. Other spaces are re-embedded
    before any destination write. Re-installing upserts by doc_id.

    CRIT-10 (match-or-re-embed): shipped vectors are used verbatim when the
    consumer's embedding model matches the pack's; on a mismatch the docs are
    re-embedded with the consumer's model instead of refusing — portable
    across models, and no dimension mismatch ever reaches the matrix build.

    CRIT-5 / authenticity boundary (§19.2): the checksum lives in the same
    zip as ``pack.db``, so it proves INTEGRITY (no accidental corruption),
    NOT AUTHENTICITY — an attacker who swaps ``pack.db`` recomputes the
    hash and rewrites the manifest. This is acceptable ONLY because v1
    installs from a LOCAL path the operator built or vetted. Do NOT add a
    remote pack-fetch path without first adding signature verification
    (§5 requires signature + checksum + pinned registry); that is a hard
    prerequisite, tracked in the design, not a nice-to-have.
    """

    pack_path = Path(pack_path)
    with tempfile.TemporaryDirectory() as staging:
        staged_db = Path(staging) / 'pack.db'
        try:
            with zipfile.ZipFile(pack_path) as zf:
                # CRIT-7: the zip is untrusted. Cap the member count and the
                # tiny manifest, and STREAM pack.db to disk in chunks with a
                # running hash + byte cap — never zf.read() an unbounded entry.
                if len(zf.namelist()) > _MAX_ZIP_MEMBERS:
                    raise SpoolError(
                        f'pack {pack_path} has {len(zf.namelist())} members '
                        f'(> {_MAX_ZIP_MEMBERS}) — refusing',
                    )
                manifest_bytes = _read_capped(
                    zf, 'manifest.yaml', max_manifest_bytes,
                )
                hasher = hashlib.sha256()
                total = 0
                with zf.open('pack.db') as src, open(staged_db, 'wb') as dst:
                    while True:
                        chunk = src.read(_INSTALL_CHUNK)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_pack_bytes:
                            raise SpoolError(
                                f'pack.db in {pack_path} is too large '
                                f'(> {max_pack_bytes} bytes) — refusing',
                            )
                        hasher.update(chunk)
                        dst.write(chunk)
        except SpoolError:
            raise
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise SpoolError(f'cannot read pack {pack_path}: {exc}') from exc

        manifest = SpoolManifest.from_dict(yaml.safe_load(manifest_bytes))
        if manifest.pack_format != PACK_FORMAT:
            raise SpoolError(
                f'pack {pack_path} declares pack format '
                f'{manifest.pack_format}; this Ariadne reads format '
                f'{PACK_FORMAT} — refusing to install',
            )

        digest = 'sha256:' + hasher.hexdigest()
        if digest != manifest.checksum:
            raise SpoolError(
                f'pack {pack_path} failed checksum verification: manifest '
                f'declares {manifest.checksum}, pack.db hashes to {digest} — '
                f'refusing to install',
            )

        # Attribution (§18.1): verify each declared license member against its
        # recorded sha256 BEFORE any store/cache write (fail-closed), staged for
        # the cache write below.
        staged_licenses = _stage_attribution_licenses(
            pack_path, manifest.attribution,
        )

        # CRIT-10 (match-or-re-embed): the pack's vectors are only meaningful
        # under the model that produced them. If the consumer's embedding
        # model MATCHES, use the shipped vectors verbatim (the perf win). If
        # it DIFFERS (dimension or model), RE-EMBED the docs' content with the
        # consumer's model instead of refusing — so a pack is portable across
        # models and a dimension mismatch can never reach the matrix build.
        from library.model_binding import inspect_binding
        binding = inspect_binding(library.path)
        consumer_identity = binding.manifest['embedding'] if binding else None
        # Open a disposable copy because schema migrations must not mutate
        # the checksum-verified artifact later written to the cache.
        working_db = Path(staging) / 'working-pack.db'
        shutil.copyfile(staged_db, working_db)

        # Stage all inference first. One transaction covers documents and
        # derived metadata, including reinstalls. The cache carries a pending
        # marker until commit so a process exit cannot register partial work.
        # CRIT-9: land docs under the reserved ``spool:<env>`` source id, not
        # the raw environment name — so a spool can never co-mingle with a
        # real source of the same name, and uninstall/disable are clean.
        install_source = spool_source_id(manifest.environment)
        with _artifact_library(working_db) as pack:
            pack_docs = pack.list_documents()
            theme_ids = _check_install_ownership(library, pack, pack_docs, manifest.environment)
            reembed = _needs_reembedding(manifest, consumer_identity, _embedding_dimension(pack))
            records = _stage_records(pack, pack_docs, library, reembed=reembed, pack_path=pack_path)
            with _staged_cache(cache_dir, manifest.environment, manifest_bytes,
                               staged_db, staged_licenses) as publish_cache:
                with library._conn_provider.acquire() as connection:
                    connection.execute('BEGIN IMMEDIATE')
                    for doc, chunks, sections in records.values():
                        _copy_doc_with_sections(
                            library, doc, embedding=doc.embedding,
                            metadata=dict(doc.metadata), source_name=None if doc.id in theme_ids else install_source,
                            sections=sections, chunks=chunks,
                            allow_reserved_source=True, connection=connection,
                        )
                    # Every row, including existing content and derived metadata,
                    # participates in the same transaction on install/reinstall.
                    _copy_source_scip(pack, library, manifest.environment, dest_source_name=install_source)
                    _copy_version_facts(pack, library, manifest.environment)
                    _copy_surface_tags(pack, library, manifest.environment)
                    _copy_spool_themes(pack, library, manifest.environment, copy_documents=False)
                    publish_cache()

    return manifest


def uninstall_pack(library, environment: str, *, cache_dir) -> int:
    """Physically remove an installed spool (CRIT-9 clean reversibility).

    Deletes exactly the reserved ``spool:<environment>`` rows — never a
    real source — and removes the cache dir. Returns the doc count removed.
    Idempotent: uninstalling an absent spool removes nothing and returns 0.
    """
    source_id = spool_source_id(environment)
    doc_ids = [
        meta.id for meta in library.list_documents_lite()
        if meta.source_name == source_id
    ]
    for doc_id in doc_ids:
        library.delete_document(doc_id)
    shutil.rmtree(Path(cache_dir) / environment, ignore_errors=True)
    return len(doc_ids)
