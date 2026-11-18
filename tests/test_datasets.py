from ielts_tutor.datasets import fix_mojibake, official_overall, stratified_holdout
from ielts_tutor.rag.schema import ScoredSample


def test_fix_mojibake_repairs_curly_quotes_and_keeps_clean_text():
    assert fix_mojibake("individualâ\x80\x99s need") == "individual’s need"
    assert fix_mojibake("plain text, café") == "plain text, café"


def test_official_overall_rounding():
    assert official_overall([6, 6, 6, 6]) == 6.0
    assert official_overall([6, 6, 6, 7]) == 6.5  # mean 6.25 rounds up to 6.5
    assert official_overall([6.5, 6.5, 6.5, 7]) == 6.5  # mean 6.625 stays 6.5
    assert official_overall([7, 7, 7, 6]) == 7.0  # mean 6.75 rounds up to 7


def test_stratified_holdout_keeps_bands_and_is_disjoint():
    samples = [
        ScoredSample(sample_id=f"s{i}", skill="writing", task="task2", response_text=f"essay {i}",
                     scores={"overall": 5.5 if i % 2 else 7.0})
        for i in range(40)
    ]
    keep, hold = stratified_holdout(samples, 0.25, seed=1)
    assert len(hold) == 10 and len(keep) == 30
    assert not {s.sample_id for s in keep} & {s.sample_id for s in hold}
    assert {s.scores["overall"] for s in hold} == {5.5, 7.0}
