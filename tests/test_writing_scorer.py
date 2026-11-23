import numpy as np

from ielts_tutor.importers.textquality import Vocab
from ielts_tutor.writing.evaluate import fit_weights
from ielts_tutor.writing.features import extract
from ielts_tutor.writing.scorer import overall_band, qwk, round_half


def test_overall_band_follows_ielts_rounding():
    assert overall_band({"TR": 6, "CC": 6, "LR": 6, "GRA": 6.5}) == 6.0  # 6.125
    assert overall_band({"TR": 6, "CC": 6, "LR": 6.5, "GRA": 6.5}) == 6.5  # 6.25 -> 6.5
    assert overall_band({"TR": 6.5, "CC": 6.5, "LR": 7, "GRA": 7}) == 7.0  # 6.75 -> 7
    assert round_half(6.74) == 6.5 and round_half(9.7) == 9.0


def test_qwk_perfect_and_off_by_one():
    y = [5, 5.5, 6, 6.5, 7, 7.5, 8]
    assert qwk(y, y) == 1.0
    assert 0.8 < qwk(y, [v + 0.5 for v in y]) < 1.0
    assert qwk(y, y[::-1]) < 0


def test_features_count_linkers_paragraphs_and_spelling():
    essay = ("Firstly, cities are crowded. However, people still move there.\n\n"
             "Moreover, jobs are concentrated in cities. In conclusion, I agre with this view.")
    f = extract(essay, "Why do people move to cities?", "task2", Vocab(set(
        "firstly cities are crowded however people still move there moreover jobs concentrated in conclusion i with "
        "this view".split())))
    assert f.paragraphs == 2 and f.sentences == 4
    assert f.linker_types == 4  # firstly, however, moreover, in conclusion
    assert f.unknown_word_ratio > 0  # "agre"
    assert f.prompt_overlap > 0.5 and f.words_vs_min < 0.2


def test_fit_weights_prefers_the_accurate_source():
    y = np.array([5, 6, 7, 8, 6.5, 5.5])
    w = fit_weights(y + 0.05, np.full(6, 6.5), y + np.array([1, -1, 1, -1, 1, -1]), y)
    assert w[0] >= 0.9
