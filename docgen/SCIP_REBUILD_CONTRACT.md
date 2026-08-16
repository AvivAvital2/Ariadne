# The contract the rebuilt modules must satisfy

Extracted mechanically from every importer in the repo. 4 modules are being
rewritten; the consumers are not touched, so every name below must exist with
the same behaviour.


## docgen/scip_cross_source.py

- `CrossSourceEdge` — 3 importer(s): cli/callers.py, tests/test_catalog_enrich_scip.py, tests/test_cross_source_edge_index.py
- `CrossSourceGraph` — 37 importer(s): cli/callers.py, cli/generation.py, docgen/catalog_enrich.py, docgen/orchestrator.py …
- `CrossSourceSymbol` — 5 importer(s): tests/test_catalog_enrich_scip.py, tests/test_cross_source_edge_index.py, tests/test_data_layer_budget.py, tests/test_scip_descriptors.py …
- `DEFAULT_ASSERT_MIN_CONFIDENCE` — 1 importer(s): tests/test_sql_assert_min_confidence.py
- `ImpactReport` — 1 importer(s): cli/callers.py
- `ReachSite` — 1 importer(s): tests/test_scip_external_resolution.py
- `SharedDatabase` — 1 importer(s): tests/test_cross_source_shared_db.py
- `_CONFIDENCE_RANK` — 3 importer(s): docgen/sql_query_views.py, docgen/sql_schema.py, tests/test_sql_assert_min_confidence.py
- `_floor_rank` — 2 importer(s): docgen/sql_query_views.py, tests/test_sql_assert_min_confidence.py
- `build_reach_findings` — 1 importer(s): tests/test_scip_external_resolution.py
- `classify_edge` — 2 importer(s): docgen/trace_flow.py, tests/test_scip_edge_typing.py
- `compute_impact_radius` — 2 importer(s): cli/callers.py, tests/test_impact_radius_cross_language.py
- `load_source_from_manifest` — 4 importer(s): docgen/reverse_augment.py, docgen/scip_persist.py, tests/test_scip_cross_source.py, tests/test_scip_manifest_source_root.py
- `shared_databases_from_config` — 2 importer(s): cli/callers.py, tests/test_cross_source_shared_db.py

## docgen/scip_extractor.py

- `ScipIndex` — 66 importer(s): docgen/orm_bindings/access.py, docgen/orm_bindings/django.py, docgen/orm_bindings/engine.py, docgen/orm_bindings/slick.py …
- `_ScipDoc` — 55 importer(s): docgen/scip_akka_http_extractor.py, docgen/scip_cross_source.py, docgen/scip_express_route_extractor.py, docgen/scip_go_http_client_extractor.py …
- `_ScipOccurrence` — 51 importer(s): docgen/scip_akka_http_extractor.py, docgen/scip_cross_source.py, docgen/scip_express_route_extractor.py, docgen/scip_go_http_client_extractor.py …
- `_ScipSymbol` — 24 importer(s): tests/test_cmd_callers.py, tests/test_cmd_index_typescript_e2e.py, tests/test_extract_elements_scip_dispatch.py, tests/test_generated_docs_attribution.py …
- `_proto_to_doc` — 1 importer(s): tests/test_owning_enclosing_range.py
- `_qualified_name_from_symbol` — 1 importer(s): docgen/scip_scala_test_extractor.py
- `_subtype` — 1 importer(s): tests/test_go_catalog_extraction.py
- `apply_vue_mapping` — 3 importer(s): docgen/scip_config.py, docgen/scip_cross_source.py, tests/test_owning_enclosing_range.py
- `extract` — 4 importer(s): docgen/catalog_extractor.py, tests/test_go_catalog_extraction.py, tests/test_scalatest_extraction.py, tests/test_scip_extractor.py

## docgen/scip_freshness.py

- `absolute_from_scip` — 2 importer(s): library/debug.py, tests/test_scip_freshness.py
- `changed_indexed_files` — 1 importer(s): tests/test_scip_freshness.py
- `freshness_warning` — 2 importer(s): cli/callers.py, library/intelligence.py
- `resolve_scip_file` — 3 importer(s): library/debug.py, library/intelligence.py, tests/test_scip_freshness.py
- `source_for_file` — 1 importer(s): tests/test_scip_freshness.py
- `stale_report` — 2 importer(s): cli/callers.py, tests/test_scip_freshness.py
- `stale_report_for_files` — 2 importer(s): library/intelligence.py, tests/test_scip_freshness.py

## docgen/scip_persist.py

- `dangling_autodoc` — 1 importer(s): cli/generation.py
- `persist_akka_http_endpoints` — 2 importer(s): cli/index.py, tests/test_persist_akka_http_endpoints.py
- `persist_all_sources` — 3 importer(s): cli/index.py, tests/test_persist_all_sources.py, tests/test_persist_progress.py
- `persist_api_endpoints` — 2 importer(s): cli/index.py, tests/test_persist_api_endpoints.py
- `persist_config_reads` — 2 importer(s): cli/index.py, tests/test_persist_config_reads.py
- `persist_config_values` — 2 importer(s): cli/index.py, tests/test_persist_config_values.py
- `persist_data_model` — 4 importer(s): cli/index.py, tests/test_data_model_integrity.py, tests/test_persist_progress.py, tests/test_sql_data_model_wiring.py
- `persist_express_routes` — 2 importer(s): cli/index.py, tests/test_persist_express_routes.py
- `persist_go_http_clients` — 2 importer(s): cli/index.py, tests/test_persist_go_extractors.py
- `persist_go_routes` — 2 importer(s): cli/index.py, tests/test_persist_go_extractors.py
- `persist_js_http_clients` — 2 importer(s): cli/index.py, tests/test_persist_js_http_clients.py
- `persist_literal_url_clients` — 1 importer(s): cli/index.py
- `persist_python_http_clients` — 2 importer(s): cli/index.py, tests/test_persist_python_http_clients.py
- `persist_python_routes` — 2 importer(s): cli/index.py, tests/test_persist_python_routes.py
- `persist_scala_http_clients` — 2 importer(s): cli/index.py, tests/test_persist_scala_http_clients.py
- `persist_string_literals` — 3 importer(s): cli/index.py, tests/test_persist_akka_http_endpoints.py, tests/test_persist_progress.py
- `persist_url_resolver` — 2 importer(s): cli/index.py, tests/test_persist_url_resolver.py
