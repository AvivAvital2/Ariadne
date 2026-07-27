"""Spool pack container — build and install.

A pack is ONE zip: ``manifest.yaml`` (whose ``checksum`` covers the DB
inside) + ``pack.db`` — a genuine Ariadne library DB holding exactly the
spool source's documents, embeddings included, so installing never
re-embeds (design §9 knowledge pack · §17 storage · §18.6.1/.2 tagging).
"""
import hashlib
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import yaml

from config import OFFICIAL_DOC_PROVENANCE
from docgen.extraction_coverage import EXTRACTION_COVERAGE_VERSION
from spools import SpoolError, SpoolManifest, spool_source_id

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


def _copy_doc_with_sections(dest, doc, *, embedding, metadata, source_name,
                            sections, allow_reserved_source=False):
    """Copy one document (+ its sections) into ``dest`` — the shared body of
    the build and install copy loops, differing only in these values.

    ``allow_reserved_source`` is the install side's privilege to write under the
    reserved ``spool:`` namespace (post-verification); the build side leaves it
    False (it writes the plain environment source name)."""
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
        # The remaining source-scoped SCIP layers, copied uniformly
        # (runs even when the corpus has no symbols).
        for table in _SCIP_SOURCE_SCOPED_LAYERS:
            _copy_source_scoped_table(
                sconn, dconn, table, source_name, target_source,
                strip_path_prefix=strip_path_prefix,
            )
        dconn.commit()


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
        for entry in sorted(repo_dir.iterdir()):
            low = entry.name.lower()
            if not entry.is_file() or not any(
                    low == n or low.startswith(n + '.') or low.startswith(n + '-')
                    for n in _ATTRIBUTION_FILENAMES):
                continue
            data = entry.read_bytes()
            blobs[f'licenses/{repo}/{entry.name}'] = data
            files.append({'name': entry.name, 'sha256': _sha256(data)})
        if files:
            records.append({'repo': repo, 'sha': sha, 'files': tuple(files)})
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
) -> SpoolManifest:
    """Build a pack zip from ``library``'s ``environment`` source.

    The environment id IS the source name — docs are selected by
    ``source_name == environment``, so the installed pack lands under the
    same name ``resolve_spools`` registers (scope-union consistency).
    Certified-dir docs are tagged ``official`` in the PACK only; the
    building store is never mutated.
    """
    from library import Library

    # HIGH-3: read ONLY the target source's docs — never
    # ``list_documents()`` (which materializes every source's content +
    # embeddings into memory; fatal on a real multi-GB store). Lite metas
    # are light; full docs are fetched in bounded batches for our ids only.
    ids = [
        meta.id for meta in library.list_documents_lite()
        if meta.source_name == environment
    ]
    if not ids:
        raise SpoolError(
            f'no documents for source {environment!r} — nothing to pack',
        )
    source_root = Path(source_root).resolve()
    # Certification is judged on the ORIGINAL absolute paths below; every
    # written row is then rewritten corpus-relative (_strip_builder_paths).
    root_prefix = str(source_root) + '/'

    import numpy as np

    embedding_dim: int | None = None
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / 'pack.db'
        with Library(db_path) as pack:
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
                    # CRIT-10: capture the embedding dim off the first real
                    # vector in this single pass (no second full scan).
                    if embedding_dim is None and doc.embedding is not None:
                        embedding_dim = int(
                            np.asarray(doc.embedding, dtype=np.float32).shape[0],
                        )
                    shipped = _strip_builder_paths(doc, root_prefix)
                    _copy_doc_with_sections(
                        pack, shipped, embedding=doc.embedding,
                        metadata=metadata,
                        source_name=doc.source_name,
                        sections=sections.get(doc.id),  # HIGH-2: sections travel
                    )
            _copy_source_scip(library, pack, environment,
                              strip_path_prefix=root_prefix)
        _checkpoint(db_path)
        db_blob = db_path.read_bytes()

    # CRIT-10: stamp the embedding identity so install can verify it. The
    # dimension came off the actual vectors above; the model is the builder's
    # configured embedding model. None when the corpus carries no embeddings.
    import embedding as _embedding
    embedding_model = _embedding.DEFAULT_MODEL if embedding_dim is not None else None

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
        corpus_shas=dict(corpus_shas or {}),
        taxonomy=tuple(taxonomy or ()),
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
    Embeddings travel as stored blobs; installing never calls an
    embedding API. Idempotent: re-installing upserts by doc_id.

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
    from library import Library

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
        reembed = False
        if manifest.embedding_dim is not None:
            import embedding
            reembed = (
                manifest.embedding_dim != embedding.EMBEDDING_DIM
                or (manifest.embedding_model
                    and manifest.embedding_model != embedding.DEFAULT_MODEL)
            )

        # CRIT-4: copy the docs into the store FIRST, and only write the cache
        # files (which register the spool) AFTER the copy fully succeeds — so a
        # crash mid-copy leaves the spool UNREGISTERED (honest "not installed"),
        # never a partial store that reports healthy. On failure, roll back the
        # rows THIS install newly created (a reinstall's pre-existing rows are
        # left intact — the upsert is idempotent).
        # CRIT-9: land docs under the reserved ``spool:<env>`` source id, not
        # the raw environment name — so a spool can never co-mingle with a
        # real source of the same name, and uninstall/disable are clean.
        install_source = spool_source_id(manifest.environment)
        newly_inserted: list[str] = []
        try:
            with Library(staged_db) as pack:
                pack_docs = pack.list_documents()
                # CRIT-10 fallback: on a model mismatch, re-embed the docs
                # that carried a vector — with the consumer's model — so the
                # stored vectors live in the consumer's space. Docs that had
                # no embedding stay unembedded. On match, keep shipped vectors.
                embeddings = {d.id: d.embedding for d in pack_docs}
                if reembed:
                    from embedding import doc_embedding_text, embed_batch_sync
                    to_embed = [d for d in pack_docs if d.embedding is not None]
                    if to_embed:
                        # Embed over the CANONICAL doc text (shared helper),
                        # so re-embedded spool vectors match how the consumer
                        # embeds its own docs — content-only would skew ranks.
                        try:
                            vecs = embed_batch_sync(
                                [doc_embedding_text(d.title, d.content)
                                 for d in to_embed],
                            )
                        except Exception as exc:
                            raise SpoolError(
                                f'pack {pack_path}: re-embedding under the '
                                f'consumer model failed ({exc}); ensure the '
                                f'embedding API is configured',
                            ) from exc
                        if len(vecs) != len(to_embed):
                            raise SpoolError(
                                f'pack {pack_path}: re-embed returned '
                                f'{len(vecs)} vectors for {len(to_embed)} docs '
                                f'— refusing (would misalign embeddings)',
                            )
                        embeddings.update(
                            {d.id: v for d, v in zip(to_embed, vecs)},
                        )
                for doc in pack_docs:
                    if library.get_document(doc.id) is None:
                        newly_inserted.append(doc.id)
                    _copy_doc_with_sections(
                        library, doc, embedding=embeddings[doc.id],
                        metadata=dict(doc.metadata), source_name=install_source,
                        sections=pack.get_sections(doc.id),
                        allow_reserved_source=True,
                    )
                # Land the environment's self-contained SCIP graph under the
                # same reserved namespace as the docs (canonical_ids raw).
                _copy_source_scip(
                    pack, library, manifest.environment,
                    dest_source_name=install_source,
                )
        except Exception:
            for doc_id in newly_inserted:
                library.delete_document(doc_id)
            raise

        dest = Path(cache_dir) / manifest.environment
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'manifest.yaml').write_bytes(manifest_bytes)
        shutil.copyfile(staged_db, dest / 'pack.db')
        for (repo, name), blob in staged_licenses.items():
            lic_dir = dest / 'licenses' / repo
            lic_dir.mkdir(parents=True, exist_ok=True)
            (lic_dir / name).write_bytes(blob)
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
