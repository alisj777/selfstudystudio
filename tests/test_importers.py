import pytest

from ielts_tutor.importers.ielts_org import parse_speaking, parse_writing

WRITING_PAGE = """
Sample Academic Writing Part {part}
Candidate Response {n}
The chart shows how children travelled
to school in 1990 and 2010.
Examiner comment
Band {band}
Key features are covered.
"""

SPEAKING = """
Part 1 – Introduction and interview: Examiner Frame
Let's talk about your home town or village.
1. What kind of place is it?
2. Would you say it's a good place to live? (Why?)
Part 1 – Introduction and interview: Recording
Describe something you own which is very important to you.
You should say:
where you got it from
and explain why it is important to you.
Part 3 – Two-way Discussion: Examiner Frame
1. Do you think advertising influences what people buy?
Part 3 – Two-way Discussion: Recording
"""


def test_parse_writing_reads_band_comment_and_task():
    pages = [WRITING_PAGE.format(part=p, n=n, band=b) for p, n, b in [(1, 1, "6"), (1, 2, "4"), (2, 1, "5.5"), (2, 2, "7.5")]]
    samples = parse_writing(["cover page"] + pages)
    assert [s.scores["overall"] for s in samples] == [6.0, 4.0, 5.5, 7.5]
    assert [s.task for s in samples] == ["task1", "task1", "task2", "task2"]
    assert samples[0].rater == "official"
    assert samples[0].response_text.startswith("The chart shows how children travelled to school")
    assert samples[0].comments == "Key features are covered."


def test_parse_writing_fails_loudly_on_layout_change():
    with pytest.raises(ValueError):
        parse_writing(["nothing here"])


def test_parse_speaking_extracts_three_parts():
    chunks = parse_speaking([SPEAKING])
    assert [c.question_type for c in chunks] == ["qt.speaking.part1", "qt.speaking.part2", "qt.speaking.part3"]
    assert "Would you say it's a good place to live?" in chunks[0].text
    assert chunks[1].text.startswith("Describe something you own")
    assert all(c.source_type == "official_sample" for c in chunks)
