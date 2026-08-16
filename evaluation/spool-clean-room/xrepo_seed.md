# Cross-repo question seed — Databricks spool (spark ↔ delta ↔ databricks-sdk-py)

Goal: questions that require BOTH (a) connecting ≥2 repos through a real architectural
link, AND (b) intricate, this-corpus code knowledge (exact classes/methods/fields/rule
ordering/divergences) — NOT answerable from general Spark/Delta lore, and NOT answerable
by grepping one repo because the answer spans the delta→spark delegation boundary.

## The three repos (corpus-relative prefixes)
- `spark/`  — Apache Spark 4.0 (Catalyst, execution, streaming, connector API, Spark Connect)
- `delta/`  — Delta Lake 4.0. TWO sub-parts:
    - Spark **connector**: `org.apache.spark.sql.delta.*`, `io.delta.tables.*`, `io.delta.sql.*`
      → GENUINELY couples to Spark. Use this for cross-repo questions.
    - **Kernel**: `io.delta.kernel.*` → standalone Java, REIMPLEMENTS Spark concepts.
      Only use for "where does the connector bridge Kernel ↔ Catalyst" questions.
- `databricks-sdk-py/` — standalone REST client. NO code calls into spark/delta.
  Only CONCEPTUAL runtime links (how an SDK job/cluster/warehouse config maps to a Spark
  runtime behavior). Keep to a small handful; these are weaker.

## Quality bar
- BAD (general, a bare LLM answers from memory): "How does Delta achieve ACID?"
- BAD (single-repo): "What does ClassicMergeExecutor do?"
- GOOD (cross-repo + intricate): "When Delta's `ClassicMergeExecutor.writeAllChanges`
  builds the merge output DataFrame, which Spark Catalyst construct does it rely on to
  evaluate the per-row match instructions, and how does that path differ from Spark-native
  SQL MERGE's `MergeRowsExec`?" — hinges on a specific delta method AND a specific spark
  class AND their divergence.
- Vary phrasing: not every question starts "How does". Use "Trace…", "Which…", "When…,
  what…", "Where does…", "Why does…diverge from…", "What breaks if…".

## Connection families and grounded bridge classes (from the pack's coupling themes)

A. Row-level DML (MERGE/UPDATE/DELETE)
   delta: io.delta.tables.DeltaMergeBuilder, DeltaMergeInto, PreprocessTableMerge,
          ClassicMergeExecutor, ResolveDeltaMergeInto, DeltaOperations.Merge
   spark: MergeIntoWriter, MergeRowsExec, RewriteMergeIntoTable,
          ResolveRowLevelCommandAssignments, ReplaceData/WriteDelta, MergeAction

B. Write path / file output
   delta: TransactionalWrite.writeFiles, DeltaFileFormatWriter, DelayedCommitProtocol,
          Snapshot, OptimisticTransaction.commit
   spark: FileFormatWriter, FileFormat/OutputWriter, FileCommitProtocol,
          BasicWriteJobStatsTracker, WriteJobDescription

C. DataSource V2 catalog/table integration
   delta: DeltaCatalog, DeltaTableV2, WriteIntoDelta, DeltaDataSource, DeltaTableV2.toBaseRelation
   spark: TableCatalog, SupportsWrite/SupportsRead, V2SessionCatalog, DataSourceV2Strategy,
          org.apache.spark.sql.connector.catalog.Table, connector.write.Write, connector.read.Scan/Batch

D. Data skipping / stats / predicate pushdown
   delta: DataSkippingReader, PrepareDeltaScan, DeltaScanGenerator, column stats (StatisticsCollection)
   spark: Catalyst Expression/Predicate, SupportsPushDownFilters, DataSourceStrategy.selectFilters,
          FileSourceStrategy

E. Structured Streaming
   delta: DeltaSink, DeltaSource, DeltaSourceOffset, DeltaSourceBase, SnapshotManagement
   spark: MicroBatchExecution, Sink/Source, Offset, StreamExecution, StateStore, OffsetSeqLog

F. Generated / Identity / Default columns
   delta: GeneratedColumn, IdentityColumn, DeltaSqlParser
   spark: ResolveDefaultColumns, connector.write, Catalyst analysis rules, ColumnDefaultValue

G. Schema & type reconciliation
   delta connector: SchemaMergingUtils, SchemaUtils, DeltaOptions
   spark: StructType/DataType, Cast/type coercion, TableOutputResolver, ResolveOutputRelation
   (Kernel bridge only: io.delta.kernel.types.* ↔ Catalyst via the connector's conversion)

H. OPTIMIZE / Z-order / compaction / repartition
   delta: OptimizeExecutor, ZCubeInfo, DeltaOptimizeBuilder, OptimizeMetrics
   spark: RepartitionByExpression, range/hash partitioning, RebalancePartitions, ShuffleExchangeExec

I. Spark Connect ↔ Delta Connect
   delta: delta.python.delta.connect.plan.* (MergeAction, UpdateTable, Vacuum, Generate),
          DeltaTable.merge (connect client), DeltaMergeBuilder (connect)
   spark: SparkConnectPlanner, proto Relation/Command, MergeIntoWriter (connect), relations.proto

J. Kernel ↔ Catalyst bridge (parallel systems meeting point)
   delta kernel: io.delta.kernel.expressions.{Predicate,And,Or,Column}, kernel.types.*, kernel.engine.Engine
   spark: Catalyst Expression, DefaultEngine expression conversion, connector predicate pushdown

K. Vectorized Parquet read + deletion vectors
   delta: DeltaParquetFileFormat, deletion-vector-aware reader, DeletionVectorDescriptor
   spark: VectorizedParquetRecordReader, ColumnarBatch/ColumnVector, ParquetFileFormat, SupportsColumnarReads

L. LogStore / filesystem
   delta: LogStore, DelegatingLogStore, HDFSLogStore/GCSLogStore/AzureLogStore
   spark: SparkPath, Hadoop Path/FileSystem, Utils

M. (few) SDK → runtime conceptual
   sdk: JobsExt (w.jobs), clusters, warehouses, DbfsExt, config/auth
   runtime: how a submitted task's cluster/runtime maps to Spark execution; NOT code-coupled.
