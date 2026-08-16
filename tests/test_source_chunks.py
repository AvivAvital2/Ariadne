"""Deterministic exact-source chunks: causal spans with stable X placeholders.

The formulation model must never retype source. Chunks carry the exact bytes and
coordinates of the causal statements inside selected definition bodies; narration
references ``{{X#}}`` and deterministic expansion re-attaches the code afterwards.

Synthetic fixtures only: source ``src1``, file ``pkg/flow.scala``.
"""
from __future__ import annotations

import pytest

from library.source_chunks import (
    derive_source_chunks,
    expand_source_placeholders,
    render_source_ledger,
)
from library.source_materialization import SourceExcerpt

SOURCE = "src1"
FILE = "pkg/flow.scala"

BODY_LINES = (
    "def writeChanges(spark: Session, txn: Txn): Seq[Action] = {",  # 100 signature
    "  val schema = target.schema()",                               # 101 unused
    '  log.info("starting write")',                                 # 102 noise
    "  val iterator = new RowIterator(source.rows)",                # 103 feeds 104
    "  val filtered = iterator.filter(row != null)",                # 104 seed
    "  if (mode == InsertOnly) {",                                  # 105 predicate
    '    metrics.record("insert-only")',                            # 106 noise
    "    val output = filtered",                                    # 107 feeds 110
    "      .select(outputCols: _*)",                                # 108 chained
    "      .filter(keepFlag == false)",                             # 109 chained
    "    writer.writeFiles(",                                       # 110 seed
    "      spark, txn, output)",                                    # 111 continuation
    "  }",                                                          # 112
    "  txn.commit(actions)",                                        # 113 causal terminal
    "}",                                                            # 114
)


def _body(lines=BODY_LINES, *, line_start=100, file=FILE, sha256="bodysha"):
    return SourceExcerpt(
        source_name=SOURCE, file=file, line_start=line_start,
        line_end=line_start + len(lines) - 1, kind="definition_body",
        content="\n".join(lines), sha256=sha256)
def test_causal_chunks_cover_seeded_statements_with_exact_coordinates():
    """Seeds pull in their whole statement, their dataflow, and their branch guard.

    The multiline call keeps its continuation line, the chained assignment keeps
    every ``.`` line, the constructor feeding the seeded filter rides along, the
    enclosing branch predicate is retained, and the final expression consuming a
    seeded identifier — the transaction commit — is the chain's product. Unused
    assignments, logging, and metrics stay out. Text and coordinates are the
    body's exact bytes; the hash is the body's hash; a second derivation is
    identical.
    """
    body = _body()
    seeds = ((FILE, 104), (FILE, 110))

    chunks = derive_source_chunks((body,), seeds)

    assert [chunk.id for chunk in chunks] == ["X1", "X2", "X3", "X4"]
    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (100, 100), (103, 105), (107, 111), (113, 113)]
    assert chunks[1].lines == (
        (103, "  val iterator = new RowIterator(source.rows)"),
        (104, "  val filtered = iterator.filter(row != null)"),
        (105, "  if (mode == InsertOnly) {"),
    )
    assert chunks[2].lines == (
        (107, "    val output = filtered"),
        (108, "      .select(outputCols: _*)"),
        (109, "      .filter(keepFlag == false)"),
        (110, "    writer.writeFiles("),
        (111, "      spark, txn, output)"),
    )
    assert chunks[3].lines == ((113, "  txn.commit(actions)"),)
    assert "result" in chunks[3].reason
    rendered = "\n".join(text for chunk in chunks for _, text in chunk.lines)
    assert "log.info" not in rendered
    assert "metrics.record" not in rendered
    assert "val schema" not in rendered
    assert all(chunk.sha256 == "bodysha" for chunk in chunks)
    assert all(chunk.file == FILE and chunk.source_name == SOURCE
               for chunk in chunks)
    assert derive_source_chunks((body,), seeds) == chunks
def test_duplicate_bodies_and_seeds_emit_each_chunk_once():
    """The same body reached by two hops, seeded twice, is still one ledger."""
    body = _body()
    chunks = derive_source_chunks(
        (body, body), ((FILE, 110), (FILE, 110), (FILE, 104)))

    assert [chunk.id for chunk in chunks] == ["X1", "X2", "X3", "X4"]
    assert len({(chunk.file, chunk.line_start, chunk.line_end)
                for chunk in chunks}) == 4
def test_seeds_outside_every_body_are_ignored():
    """A foreign seed adds nothing: the body behaves exactly as seedless."""
    foreign = derive_source_chunks((_body(),), (("other/file.scala", 5),))

    assert foreign == derive_source_chunks((_body(),), ())
    assert [(chunk.line_start, chunk.line_end) for chunk in foreign] == [
        (100, 114)]
    assert foreign[0].reason == "short_body"


def test_unisolatable_statement_retains_the_whole_body_and_says_why():
    """A statement whose brackets never balance cannot be isolated safely."""
    body = _body((
        "def broken(a: Int) = {",
        "  call(unclosed(",
        "}",
    ), line_start=200)

    chunks = derive_source_chunks((body,), ((FILE, 201),))

    assert len(chunks) == 1
    assert (chunks[0].line_start, chunks[0].line_end) == (200, 202)
    assert chunks[0].reason == "unisolatable"
    assert chunks[0].lines[1] == (201, "  call(unclosed(")
def test_expansion_is_exact_deterministic_and_rejects_unknown_ids():
    """{{X#}} becomes ``file:line `exact text``` lines; nothing is retyped."""
    chunks = derive_source_chunks((_body(),), ((FILE, 104), (FILE, 110)))
    answer = "The write is proven by:\n{{X3}}\nfed by the bare-id chunk\nX2\n(done)"

    expanded = expand_source_placeholders(answer, chunks)

    assert f"{FILE}:110 `    writer.writeFiles(`" in expanded
    assert f"{FILE}:111 `      spark, txn, output)`" in expanded
    assert f"{FILE}:103 `  val iterator = new RowIterator(source.rows)`" in expanded
    by_line = dict(line for chunk in chunks for line in chunk.lines)
    code_lines = 0
    for rendered in expanded.splitlines():
        if "`" not in rendered:
            continue
        code_lines += 1
        coordinate, text = rendered.split(" `", 1)
        file, line = coordinate.rsplit(":", 1)
        assert file == FILE
        assert text[:-1] == by_line[int(line)]
    assert code_lines == len(chunks[1].lines) + len(chunks[2].lines)
    assert expand_source_placeholders(answer, chunks) == expanded
    with pytest.raises(ValueError, match="X9"):
        expand_source_placeholders("see {{X9}}", chunks)
    assert "[unsupported evidence X9]" in expand_source_placeholders(
        "see {{X9}}", chunks, strict=False)


def test_ledger_renders_each_chunk_once_with_its_coordinates():
    chunks = derive_source_chunks((_body(),), ((FILE, 104), (FILE, 110)))

    ledger = render_source_ledger(chunks)

    assert ledger.count("{{X2}}") == 1
    assert f"{FILE}:103-105" in ledger
    assert "  val filtered = iterator.filter(row != null)" in ledger
    assert "log.info" not in ledger
def test_dataflow_chain_comments_and_returns_ride_the_causal_route():
    """A three-deep assignment chain, a commented seed, and the return ride along.

    The seed's trailing comment must not skew bracket balance; each assignment
    feeding the seeded call is pulled in transitively; the return of a causal
    identifier is kept while a bare ``return`` and an unrelated return stay out.
    """
    body = _body((
        "def compute(input: Data): Result = {",   # 300
        "  val alpha = load(input)",              # 301 round 3
        "  val beta = transform(alpha)",          # 302 round 2
        "  val gamma = beta",                     # 303 round 1
        "    .normalize()",                       # 304 continuation
        "  sink.push(gamma) // final push (",     # 305 seed; comment holds a (
        "  return gamma",                         # 306 causal return
        "  return",                               # 307 bare
        "  return unrelatedValue",                # 308 unrelated
        "}",                                      # 309
    ), line_start=300)

    chunks = derive_source_chunks((body,), ((FILE, 305),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (300, 306)]
    text = "\n".join(text for _, text in chunks[0].lines)
    assert "val alpha = load(input)" in text
    assert "return gamma" in text
    assert "unrelatedValue" not in text
    assert chunks[0].lines[-1] == (306, "  return gamma")
    assert "return" in chunks[0].reason


def test_predicate_walk_stays_inside_the_seeded_definition():
    """The indentation ladder never escapes into a sibling definition.

    A slice can hold two definitions; the walk from a seed in the second stops
    at the first indentation-zero line and never reports the first definition's
    branch as enclosing. Blank lines are stepped over, not tripped on.
    """
    body = _body((
        "def first(): Unit = {",          # 400
        "  if (early) {",                 # 401 sibling branch — never kept
        "    helper(1)",                  # 402
        "  }",                            # 403
        "}",                              # 404
        "def second(flag: Boolean) = {",  # 405 indent 0 — walk stops here
        "  if (flag) {",                  # 406 enclosing predicate
        "",                               # 407 blank
        "    target.run(param)",          # 408 seed
        "  }",                            # 409
        "}",                              # 410
    ), line_start=400)

    chunks = derive_source_chunks((body,), ((FILE, 408),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (400, 400), (406, 406), (408, 408)]
    rendered = "\n".join(text for chunk in chunks for _, text in chunk.lines)
    assert "early" not in rendered
    assert "helper" not in rendered


def test_empty_bodies_max_span_and_cross_body_duplicates_are_safe():
    """An empty body yields nothing; a runaway statement falls back to the whole
    body; two overlapping slices agreeing on a span emit it once."""
    empty = _body((), line_start=500)
    assert derive_source_chunks((empty,), ((FILE, 500),)) == ()

    runaway = _body((
        "def long(): Unit = {",
        "  build(",
        "    one,",
        "    two,",
        "    three,",
        "  )",
        "}",
    ), line_start=600)
    fallback = derive_source_chunks((runaway,), ((FILE, 601),),
                                    max_statement_span=3)
    assert len(fallback) == 1
    assert fallback[0].reason == "unisolatable"
    assert (fallback[0].line_start, fallback[0].line_end) == (600, 606)

    wide = _body((
        "    val output = filtered",   # 107
        "      .select(outputCols: _*)",
        "      .filter(keepFlag == false)",
        "    writer.writeFiles(",      # 110
        "      spark, txn, output)",   # 111
    ), line_start=107)
    narrow = _body((
        "    writer.writeFiles(",      # 110
        "      spark, txn, output)",   # 111
    ), line_start=110)

    chunks = derive_source_chunks((wide, narrow), ((FILE, 110),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (107, 107), (110, 111)]
    assert [chunk.id for chunk in chunks] == ["X1", "X2"]
def test_seed_on_a_chained_line_completes_its_statement_upward():
    """A call site recorded on a ``.filter`` line pulls in the statement head.

    Compiler occurrences land on the exact line of the call, which for chained
    transformations is a continuation line; the chunk carries the assignment
    head, every link of the chain above the seed, and the first statement
    consuming the chain's product.
    """
    chunks = derive_source_chunks((_body(),), ((FILE, 109),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (100, 100), (103, 105), (107, 111)]
    assert chunks[2].lines[0] == (107, "    val output = filtered")
    assert chunks[2].lines[-1] == (111, "      spark, txn, output)")
def test_final_expression_returning_the_causal_value_is_kept():
    """Expression languages return the last statement; it must ride along.

    The final code line referencing a causal identifier is retained together
    with the comment explaining it — while a final line referencing nothing
    causal stays out.
    """
    body = _body((
        "def process(input: Rows): Iterator[Row] = {",
        "  val validator = build(input)",
        "  val scanner = new RowScanner(",
        "    input, validator)",
        "  audit.log(input)",
        "  // null indicates a record must be discarded",
        "  scanner.filter(_ != null)",
        "}",
    ), line_start=700)

    chunks = derive_source_chunks((body,), ((FILE, 702),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (700, 703), (705, 706)]
    assert chunks[-1].lines == (
        (705, "  // null indicates a record must be discarded"),
        (706, "  scanner.filter(_ != null)"),
    )
    assert "result" in chunks[-1].reason
    rendered = "\n".join(text for chunk in chunks for _, text in chunk.lines)
    assert "audit.log" not in rendered

    silent = _body((
        "def finish(): Unit = {",
        "  helper(1)",
        "  metrics.flush()",
        "}",
    ), line_start=800)

    quiet = derive_source_chunks((silent,), ((FILE, 801),))

    assert [(chunk.line_start, chunk.line_end) for chunk in quiet] == [
        (800, 801)]


def test_selected_body_with_no_interior_seed_still_proves_its_content():
    """A selected body whose calls are all unindexed is still selected proof.

    Short bodies stay complete — the production slicer's convention — while a
    long seedless body contributes only its signature rather than a dump.
    """
    executor = _body((
        "def doRun(): RDD[Row] = {",
        "  child.execute().mapPartitions(process)",
        "}",
    ), line_start=800)

    chunks = derive_source_chunks((executor,), ())

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (800, 802)]
    assert chunks[0].reason == "short_body"

    tall = _body((
        "def big(): Unit = {",
        *[f"  step{index}()" for index in range(58)],
        "}",
    ), line_start=900)

    lofty = derive_source_chunks((tall,), ())

    assert [(chunk.line_start, chunk.line_end) for chunk in lofty] == [
        (900, 900)]
    assert lofty[0].reason == "signature"
def test_uncovered_call_site_excerpts_become_standalone_chunks():
    """A compiler site outside every materialized body is still exact proof.

    Its excerpt already carries the exact line, coordinates, and file hash, so
    it becomes a single-line chunk; a site covered by a selected body stays
    merged into that body's causal chunk instead of duplicating it.
    """
    body = _body()
    outside = SourceExcerpt(
        source_name=SOURCE, file="pkg/exec.scala", line_start=42, line_end=42,
        kind="call_site", content="    child.execute().mapPartitions(process)",
        sha256="execsha")
    covered = SourceExcerpt(
        source_name=SOURCE, file=FILE, line_start=110, line_end=110,
        kind="call_site", content="    writer.writeFiles(", sha256="bodysha")
    ignored = SourceExcerpt(
        source_name=SOURCE, file="pkg/exec.scala", line_start=9, line_end=9,
        kind="definition", content="def helper(): Unit = {", sha256="execsha")

    chunks = derive_source_chunks(
        (body,), ((FILE, 104), (FILE, 110)),
        sites=(outside, covered, ignored))

    spans = [(chunk.file, chunk.line_start, chunk.line_end)
             for chunk in chunks]
    assert ("pkg/exec.scala", 42, 42) in spans
    assert ("pkg/exec.scala", 9, 9) not in spans
    assert spans.count((FILE, 110, 110)) == 0
    standalone = next(
        chunk for chunk in chunks if chunk.file == "pkg/exec.scala")
    assert standalone.lines == (
        (42, "    child.execute().mapPartitions(process)"),)
    assert standalone.reason == "call_site"
    assert standalone.sha256 == "execsha"
def test_comment_lines_directly_above_a_kept_line_ride_along():
    """The comment explaining a causal statement is part of its meaning.

    A contiguous comment block directly above a kept line is retained; a
    comment above an excluded statement stays out with it.
    """
    body = _body((
        "def writeAll(spark: Session): Unit = {",
        "  // Iceberg spec requires partition columns in data files",
        "  val flag = compat.isAnyEnabled(meta)",
        '  log.debug("noise")',
        "  // unrelated commentary",
        "  metrics.tick()",
        "  writer.push(flag)",
        "}",
    ), line_start=900)

    chunks = derive_source_chunks((body,), ((FILE, 906),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (900, 902), (906, 906)]
    rendered = "\n".join(text for chunk in chunks for _, text in chunk.lines)
    assert "Iceberg spec requires partition columns" in rendered
    assert "unrelated commentary" not in rendered
    assert "comment" in chunks[0].reason
def test_first_consumer_of_a_seed_produced_value_is_kept():
    """A seeded call that produces a value proves little until it is used.

    The first statement consuming the seed's assigned name is retained; noise
    between them and a non-causal final line stay out.
    """
    body = _body((
        "def resolve(field: Field): Action = {",
        "  val identityExp = IdentityColumn.create(field)",
        "  action.copy(expr = identityExp)",
        "  metrics.tick()",
        "  finish()",
        "}",
    ), line_start=900)

    chunks = derive_source_chunks((body,), ((FILE, 901),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (900, 902)]
    assert chunks[0].lines[-1] == (902, "  action.copy(expr = identityExp)")
    assert "consumer" in chunks[0].reason
    rendered = "\n".join(text for chunk in chunks for _, text in chunk.lines)
    assert "metrics.tick" not in rendered
    assert "finish()" not in rendered
def test_unisolatable_consumers_and_degenerate_sites_are_skipped_quietly():
    """Secondary evidence never destroys primary evidence.

    A consumer whose statement cannot be bounded is skipped — the seed chunk
    survives instead of the whole body collapsing to a fallback. A site
    excerpt with no content contributes nothing, and the same site offered
    twice contributes once.
    """
    body = _body((
        "def route(field: Field): Unit = {",
        "  val token = mint(field)",
        "  send(token, open(",
        "}",
    ), line_start=900)

    chunks = derive_source_chunks((body,), ((FILE, 901),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (900, 901)]
    assert "unisolatable" not in chunks[0].reason

    empty_site = SourceExcerpt(
        source_name=SOURCE, file="pkg/other.scala", line_start=5, line_end=5,
        kind="call_site", content="", sha256="othersha")
    twin = SourceExcerpt(
        source_name=SOURCE, file="pkg/other.scala", line_start=9, line_end=9,
        kind="call_site", content="  relay(token)", sha256="othersha")

    sited = derive_source_chunks(
        (body,), ((FILE, 901),), sites=(empty_site, twin, twin))

    spans = [(chunk.file, chunk.line_start) for chunk in sited]
    assert spans.count(("pkg/other.scala", 9)) == 1
    assert ("pkg/other.scala", 5) not in spans
def test_unbounded_returns_unconsumed_values_and_comment_bodies_stay_quiet():
    """The remaining degenerate shapes neither crash nor over-capture."""
    runaway_return = _body((
        "def fetch(field: Field): Row = {",
        "  val token = mint(field)",
        "  audit.keep(token)",
        "  return open(token, extra(",
        "}",
    ), line_start=970)

    chunks = derive_source_chunks((runaway_return,), ((FILE, 971),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (970, 972)]

    unconsumed = _body((
        "def park(field: Field): Unit = {",
        "  val token = mint(field)",
        "}",
    ), line_start=960)
    assert [(chunk.line_start, chunk.line_end)
            for chunk in derive_source_chunks(
                (unconsumed,), ((FILE, 961),))] == [(960, 961)]

    lonely = _body(("// commentary only",), line_start=950)
    assert [(chunk.line_start, chunk.line_end)
            for chunk in derive_source_chunks(
                (lonely,), ((FILE, 950),))] == [(950, 950)]
def test_multiline_class_signature_is_kept_through_its_body_opener():
    """A class header is one statement: parameters, extends, and the opener.

    Keeping only its first line drops the supertype list the header exists to
    state; a body whose first line already opens stays a one-line signature.
    """
    body = _body((
        "class DeltaishTable private[pkg](",
        "    val spark: Session,",
        "    val options: Map[String, String])",
        "  extends Table",
        "  with SupportsWrite {",
        "  def id: String = name(key)",
        "}",
    ), line_start=1000)

    chunks = derive_source_chunks((body,), ((FILE, 1005),))

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (1000, 1005)]
    assert any(text.strip() == "extends Table"
               for _, text in chunks[0].lines)
