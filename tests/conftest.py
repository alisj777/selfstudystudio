from pathlib import Path

import pytest

from ielts_tutor.kg import load_seed
from ielts_tutor.rag import HashingEmbedder, ScoredSample, build_indexes, multirag_from
from ielts_tutor.rag.build import load_corpus

DATA = Path(__file__).resolve().parents[1] / "data" / "ielts"


@pytest.fixture(scope="session")
def kg():
    return load_seed(DATA / "kg")


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(DATA / "corpus")


@pytest.fixture(scope="session")
def samples():
    # Synthetic fixtures for exercising retrieval only; not real scored essays.
    def s(i, overall, text):
        return ScoredSample(
            sample_id=f"t-{i}", skill="writing", task="task2", response_text=text,
            scores={"TR": overall, "CC": overall, "LR": overall, "GRA": overall, "overall": overall},
        )

    return [
        s(1, 5.0, "Online course is good because student can study at home and it is cheap."),
        s(2, 6.0, "Online learning offers flexibility, although students may miss face-to-face interaction with teachers."),
        s(3, 7.0, "While online courses provide unprecedented flexibility, they can hardly replicate the collaborative atmosphere of a campus."),
        s(4, 8.0, "Admittedly, digital platforms have democratised access to higher education; nevertheless, the tacit skills fostered by in-person seminars remain difficult to transmit online."),
        s(5, 6.5, "Traffic congestion in big cities is caused by too many private cars and poor public transport."),
    ]


@pytest.fixture(scope="session")
def indexes(kg, corpus, samples):
    return build_indexes(kg, corpus, samples, HashingEmbedder())


@pytest.fixture(scope="session")
def rag(indexes, kg):
    return multirag_from(indexes, kg)
