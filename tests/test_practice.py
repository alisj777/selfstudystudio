import json

import pytest

from ielts_tutor.importers.answer_keys import AnswerKey
from ielts_tutor.importers.cambridge import PracticeTest
from ielts_tutor.importers.explanations import ExplanationSection

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture
def setup(tmp_path):
    exam, var = tmp_path / "data", tmp_path / "var"
    pdf = tmp_path / "book.pdf"
    doc = pymupdf.open()
    for text in ("SECTION 1 Questions 1-10", "SECTION 2 Questions 11-20"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(pdf)
    mp3 = tmp_path / "t1s1.mp3"
    mp3.write_bytes(b"ID3fake")
    test = PracticeTest(test_id="cam99-t1-listening", book=99, test="1", skill="listening", pdf=str(pdf),
                        pages=[1, 2], audio={1: str(mp3)})
    keys = [AnswerKey(key_id=f"k{q}", book=99, test="1", module="both", skill="listening", question=q,
                      answers=["dogs" if q == 2 else f"a{q}"]) for q in range(1, 41)]
    expl = ExplanationSection(section_id="cam99-t1-listening-p1", book=99, test="1", skill="listening", part=1,
                              pages=[5], text="2. 题干词 dog 定位，注意复数，答案为 dogs。", source="x")
    for sub, rows in (("tests", [test]), ("answers", keys), ("explanations", [expl])):
        (exam / sub / "private").mkdir(parents=True)
        (exam / sub / "private" / "cambridge.jsonl").write_text("\n".join(r.model_dump_json() for r in rows))
    from ielts_tutor.grading.practice import PracticeService

    return PracticeService(exam, var), var


def test_catalogue_pages_audio_and_marking(setup):
    svc, var = setup
    cat = svc.catalogue("amy")
    assert cat == [{"test_id": "cam99-t1-listening", "book": 99, "test": "1", "skill": "listening", "pages": 2,
                    "audio_parts": [1], "keys": 40, "done": None}]
    png = svc.page_png("cam99-t1-listening", 1)
    assert png.read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(IndexError):
        svc.page_png("cam99-t1-listening", 2)
    with pytest.raises(KeyError):
        svc.audio_file("cam99-t1-listening", 3)
    with pytest.raises(KeyError):
        svc.get("../../etc/passwd")

    answers = {q: f"a{q}" for q in range(1, 41)}
    answers[2] = "dog"
    r = svc.mark("cam99-t1-listening", answers, "amy", seconds=1800)
    assert r["raw"] == 39 and r["band"] == 9.0
    wrong = [x for x in r["results"] if x["correct"] is False]
    assert [x["question"] for x in wrong] == [2] and "答案为 dogs" in wrong[0]["explanation"]["text"]
    profile = json.loads((var / "learners" / "amy.json").read_text())
    assert profile["history"][-1]["test_id"] == "cam99-t1-listening" and "skill.listening" in profile["mastery"]
    assert svc.catalogue("amy")[0]["done"] == 9.0
