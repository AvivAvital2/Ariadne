# Rendered charts for docs/scip-pipeline.md.
#
# The .dot files are the source of truth; the .svg files are committed so the document
# renders anywhere without a graphviz install. They live at the top level of docs/ because
# `docs/*/` is the GENERATED per-source export tree and is gitignored.
#
#     make -C docs scip-charts

DOT    := dot
CHARTS := $(patsubst %.dot,%.svg,$(wildcard scip-pipeline-*.dot))

.PHONY: scip-charts
scip-charts: $(CHARTS)

%.svg: %.dot
	$(DOT) -Tsvg -o $@ $<
	@echo "rendered $@"
