from pathlib import Path

from ielts_tutor.rag import multirag_from
from ielts_tutor.rag.embed import tokenize
from ielts_tutor.rag.evaluate import load_queries, run, summarise
from ielts_tutor.rag.glossary import Glossary

DATA = Path(__file__).resolve().parents[1] / "data" / "ielts"


def test_tokenize_stems_and_splits_hyphens():
    assert tokenize("Over-use of linking words") == tokenize("over use of linking word")


def test_glossary_matches_english_phrases_and_chinese():
    g = Glossary.load(DATA / "glossary.yaml")
    assert "cohesive_devices" in g.match("I use too many linking words")
    assert "cohesive_devices" in g.match("连接词用得太多")
    assert g.match("the weather is nice") == []


def test_glossary_expansion_adds_each_word_once():
    g = Glossary({"cd": ["linking words", "transition words", "cohesive devices"]})
    expanded = tokenize(g.expand("linking words").query)
    assert expanded.count(tokenize("words")[0]) == 1
    assert tokenize("cohesive")[0] in expanded


def test_glossary_lets_chinese_query_reach_english_rubric(indexes, kg):
    g = Glossary.load(DATA / "glossary.yaml")
    rag = multirag_from(indexes, kg, g)
    top = rag.rubric.search_descriptors("口语发音不清楚", 1, skill="speaking")[0]
    assert top.meta["criterion"] == "crit.speaking.P"


def test_eval_runs_and_summarises(indexes, kg):
    queries = load_queries(DATA / "eval" / "retrieval_queries.yaml")
    summary = summarise(run(multirag_from(indexes, kg), queries))
    assert summary["all"]["n"] == len(queries)
    assert {"rubric", "topic", "corpus", "lang:zh"} <= summary.keys()
    assert 0 <= summary["all"]["mrr"] <= 1
