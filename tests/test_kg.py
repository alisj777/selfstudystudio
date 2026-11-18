import pytest

from ielts_tutor.kg import Edge, KGError, KnowledgeGraph, Node, NodeType, Rel, validate


def test_seed_loads_without_warnings(kg):
    assert validate(kg) == []
    stats = kg.stats()
    assert stats["Criterion"] == 9
    assert stats["BandDescriptor"] == 81  # 9 criteria x bands 1-9
    assert stats["Topic"] >= 30
    assert stats["GrammarPoint"] >= 20


def test_writing_and_speaking_criteria(kg):
    writing = {c.attrs["short"] for c in kg.criteria_for("skill.writing")}
    speaking = {c.attrs["short"] for c in kg.criteria_for("skill.speaking")}
    assert writing == {"TA", "TR", "CC", "LR", "GRA"}
    assert speaking == {"FC", "LR", "GRA", "P"}


def test_every_descriptor_cites_official_source(kg):
    for d in kg.nodes(NodeType.BAND_DESCRIPTOR):
        assert d.attrs.get("source_url", "").startswith("https://ielts.org/"), d.id
        assert d.attrs.get("page") and d.attrs.get("descriptor") and d.attrs.get("key_features"), d.id


def test_descriptors_by_band(kg):
    bands = [d.attrs["band"] for d in kg.descriptors("crit.writing.GRA", [6, 7])]
    assert bands == [6, 7]


def test_topic_neighbourhood_includes_parent_and_related(kg):
    ids = {n.id for n in kg.topic_neighbourhood("topic.education.online")}
    assert "topic.education" in ids
    assert "topic.technology.internet" in ids


def test_error_types_map_to_grammar_and_criteria(kg):
    assert {g.id for g in kg.grammar_for_error("err.verb_sva")} == {"gp.svagreement"}
    assert kg.out("err.spell", Rel.AFFECTS)[0].id == "crit.writing.LR"


def test_grammar_prerequisites(kg):
    assert "gp.relative_clauses" in {g.id for g in kg.prerequisites("gp.reduced_relatives")}


def test_essay_structures_suit_task2_types(kg):
    suited = {qt.id for es in kg.nodes(NodeType.ESSAY_STRUCTURE) for qt in kg.out(es.id, Rel.SUITS)}
    for qt in kg.nodes(NodeType.QUESTION_TYPE):
        if qt.attrs.get("task") == "task2":
            assert qt.id in suited, qt.id


def test_edge_rules_are_enforced():
    g = KnowledgeGraph()
    g.add_node(Node(id="t", type=NodeType.TOPIC, name="T"))
    g.add_node(Node(id="g", type=NodeType.GRAMMAR_POINT, name="G"))
    with pytest.raises(KGError):
        g.add_edge(Edge(source="t", target="g", rel=Rel.SUBTOPIC_OF))
    with pytest.raises(KGError):
        g.add_node(Node(id="t", type=NodeType.TOPIC, name="dup"))


def test_json_round_trip_and_cypher(kg, tmp_path):
    path = tmp_path / "kg.json"
    kg.to_json(path)
    again = KnowledgeGraph.from_json(path)
    assert again.stats() == kg.stats()
    cypher = kg.to_cypher()
    assert "MERGE (:Criterion" in cypher and ":HAS_DESCRIPTOR]" in cypher
