import numpy as np

from ielts_tutor.rag import Evidence, HashingEmbedder, HybridIndex, IndexDoc, rrf_fuse
from ielts_tutor.rag.build import chunk_text, split_chunk
from ielts_tutor.rag.schema import CorpusChunk


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed(["traffic congestion in cities"]), e.embed(["traffic congestion in cities"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_index_filters_and_persistence(tmp_path):
    idx = HybridIndex(HashingEmbedder())
    idx.add([
        IndexDoc(id="a", text="online courses replace classrooms", meta={"skill": "writing", "band": 6}),
        IndexDoc(id="b", text="describe your hometown", meta={"skill": "speaking", "band": 7}),
        IndexDoc(id="c", text="online shopping habits", meta={"skill": "speaking", "band": 8}),
    ])
    speaking = [h.doc.id for h in idx.search("online", 5, {"skill": "speaking"})]
    assert speaking == ["c", "b"]  # filter keeps only speaking docs; lexical match ranks first
    assert {h.doc.id for h in idx.search("online", 5, {"band": {"gte": 6, "lte": 7}})} == {"a", "b"}
    idx.save(tmp_path / "i")
    again = HybridIndex.load(tmp_path / "i", HashingEmbedder())
    assert again.search("hometown", 1)[0].doc.id == "b"


def test_corpus_retrieval_by_topic(rag):
    hits = rag.corpus.retrieve("cars and traffic jams in big cities", 3, skill="writing")
    assert hits[0].id == "orig-w2-003"
    filtered = rag.corpus.retrieve("anything", 5, topic_ids=["topic.urban.transport"])
    assert [h.id for h in filtered] == ["orig-w2-003"]


def test_mmr_returns_distinct_items(rag):
    hits = rag.corpus.retrieve("online learning and education", 3, skill="writing", mmr_lambda=0.5)
    assert len({h.id for h in hits}) == 3


def test_near_duplicate_detection(rag):
    sim, ev = rag.corpus.max_similarity(
        "Some people believe online courses will soon replace traditional classroom teaching at universities.",
        skill="writing",
    )
    assert ev.id == "orig-w2-001" and sim > 0.7


def test_rubric_descriptor_window(rag):
    bands = [ev.meta["band"] for ev in rag.rubric.descriptors("crit.writing.CC", 6.5)]
    assert bands == [6, 7]
    bands = [ev.meta["band"] for ev in rag.rubric.descriptors("crit.writing.CC", 6)]
    assert bands == [5, 6, 7]


def test_rubric_samples_within_band_window(rag):
    hits = rag.rubric.similar_samples(
        "Online courses are flexible but students lose contact with teachers.",
        5, skill="writing", around_band=6.0,
    )
    assert hits and all(5.5 <= h.meta["overall"] <= 6.5 for h in hits)
    prior = rag.rubric.knn_band(hits)
    assert 5.5 <= prior <= 6.5


def test_topic_retrieval_expands_through_graph(rag):
    results = rag.topic.retrieve("e-learning and studying online from home", 1, skill="writing")
    assert results[0].id == "topic.education.online"
    expanded = {ev.id for ev in results[1:]}
    assert "topic.education" in expanded
    assert all(ev.meta["expanded_from"] == "topic.education.online" for ev in results[1:])


def test_multirag_fuses_sources(rag):
    results = rag.retrieve("cohesive devices and paragraphing", k=6, skill="writing")
    assert {ev.source for ev in results} >= {"rubric"}
    assert results == sorted(results, key=lambda e: e.score, reverse=True)


def test_rrf_fuse_rewards_agreement():
    a = [Evidence(source="corpus", id="x", text="", score=0), Evidence(source="corpus", id="y", text="", score=0)]
    b = [Evidence(source="corpus", id="y", text="", score=0)]
    assert [e.id for e in rrf_fuse([a, b])] == ["y", "x"]


def test_chunking_keeps_questions_whole_and_splits_passages():
    long_passage = "\n\n".join(["word " * 200] * 3)
    q = CorpusChunk(chunk_id="q", skill="writing", source_type="question_bank", text=long_passage)
    p = q.model_copy(update={"chunk_id": "p", "source_type": "passage"})
    assert len(split_chunk(q)) == 1
    parts = split_chunk(p)
    assert [c.chunk_id for c in parts] == ["p#p1", "p#p2", "p#p3"]
    assert chunk_text("a\n\nb", max_words=10) == ["a\n\nb"]
