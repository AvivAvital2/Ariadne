
from pathlib import Path


def test_offline_audit_releases_per_question_graph_before_the_next_question():
    path = (Path(__file__).parents[1] / "evaluation/chain-benchmark" /
            "offline_earliest_failure.py")
    source = path.read_text()
    assert "del artifacts, surface, story" in source
    assert "gc.collect()" in source
