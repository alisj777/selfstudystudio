from ielts_tutor.grading.objective import answer_matches, mark, parse_answer_text, raw_to_band
from ielts_tutor.importers.answer_keys import AnswerKey, assemble, parse_key_pages


def _words(rows: list[tuple[float, float, str]]) -> list[tuple]:
    """(x, y, text) rows -> PyMuPDF-style word boxes, one box per word."""
    out = []
    for x, y, text in rows:
        for w in text.split():
            out.append((x, y, x + 6 * len(w), y + 9, w))
            x += 6 * len(w) + 4
    return out


# Two columns, the way the books set them; the numbers are what the text layer gives.
PAGE = _words([
    (40, 20, "TEST 1"), (40, 40, "LISTENING"),
    (40, 60, "Section 1, Questions 1-10"), (260, 60, "Section 3, Questions 21-30"),
    *[(40, 70 + 10 * i, f"{q} {a}") for i, (q, a) in enumerate(
        [(1, "shopping / variety of shopping"), (2, "guided tours"), (3, "(free) drink(s)/refreshment(s)"),
         (4, "notice board"), (5, "13th February"), (6, "Tower of London"), (1, "Bristol"),  # "1" is a misread 7
         (8, "American Museum"), (9, "student newspaper"), (10, "Yentob")])],
    *[(260, 70 + 10 * i, f"{21 + i} {'ABCDEFGHAB'[i]}") for i in range(10)],
    (40, 180, "Section 2, Questions 11-20"), (260, 180, "Section 4, Questions 31-40"),
    (40, 190, "11 IN EITHER ORDER, BOTH REQUIRED"), (40, 200, "FOR ONE MARK"), (60, 210, "coal"), (60, 220, "firewood"),
    *[(40, 230 + 10 * i, f"{12 + i} word{12 + i}") for i in range(4)],
    (40, 270, "16&17 IN EITHER ORDER"), (60, 280, "B"), (60, 290, "E"),
    *[(40, 300 + 10 * i, f"{18 + i} word{18 + i}") for i in range(3)],
    *[(260, 190 + 10 * i, f"{31 + i} answer{31 + i}") for i in range(10)],
])


def _keys():
    return assemble(4, parse_key_pages([(153, PAGE)]))


def test_parse_two_column_key_page():
    keys = {k.question: k for k in _keys()}
    assert sorted(keys) == list(range(1, 41))
    assert keys[7].answers == ["Bristol"]  # repaired from its neighbours
    assert keys[1].answers == ["shopping", "variety of shopping"]
    assert keys[11].all_of == ["coal", "firewood"]
    assert keys[16].group == [16, 17] and keys[17].answers == ["E"]
    assert keys[25].answers == ["E"] and keys[40].answers == ["answer40"]
    assert {k.test for k in keys.values()} == {"1"} and keys[1].skill == "listening"


def test_answer_matching_rules():
    assert answer_matches("drinks", "(free) drink(s)/refreshment(s)")
    assert answer_matches("free drink", "(free) drink(s)/refreshment(s)")
    assert answer_matches("the pianist", "(the/a) pianist/piano player")
    assert not answer_matches("pianos", "(the/a) pianist/piano player")
    assert answer_matches("car park", "car-park/parking lot")
    assert answer_matches("NG", "NOT GIVEN") and answer_matches("c", "C")
    assert not answer_matches("1050", "10.50")
    assert answer_matches("thirty-five", "35 / thirty five")


def test_mark_handles_groups_and_both_required():
    keys = _keys()
    answers = {q: k.answers[0] for k in keys for q in [k.question] if k.answers}
    answers[11] = "firewood and coal"
    answers[16], answers[17] = "E", "B"  # either order
    answers[3] = "drinks"
    answers[2] = "guided tour bus"  # wrong
    m = mark(keys, answers)
    assert m.raw == 39 and m.keyed == 40
    assert [r.question for r in m.results if not r.correct] == [2]
    assert m.band == raw_to_band(39, "listening") == 9.0


def test_missing_keys_are_not_counted_and_band_is_scaled():
    keys = [AnswerKey(key_id=f"k{q}", book=1, test="1", module="academic", skill="reading", question=q,
                      answers=["TRUE"]) for q in range(1, 31)]
    m = mark(keys, {q: "TRUE" for q in range(1, 41)})
    assert m.raw == 30 and m.keyed == 30
    assert m.band == 9.0  # 30/30 scaled to 40/40
    assert raw_to_band(30, "reading", "academic") == 7.0
    assert raw_to_band(30, "reading", "general") == 6.0


def test_parse_answer_text_formats():
    assert parse_answer_text("1 litter\n2. dogs\n3) A") == {1: "litter", 2: "dogs", 3: "A"}
    assert parse_answer_text("litter, dogs, A") == {1: "litter", 2: "dogs", 3: "A"}
