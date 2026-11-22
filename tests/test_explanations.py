from ielts_tutor.importers.explanations import explanation_books, find_explanation, segment, to_records

# OCRed pages, simplified: no running headers, headings garbled the way OCR garbles them.
PAGES = {
    0: "目 录 CONTENTS\nTest 1 ...... 1\nTest 2 ...... 40",
    1: "Section I\n场景介绍\n一位学生致电询问。\n答案速查: 1.litter 2.dogs 3.insects 4.wall\n\n"
       "2. 题干词 dog 易定位，原文 dogs must be kept on a lead，注意填复数，答案为 dogs。",
    2: "Section 2\n12. 本题定位词 bus，原文 the bus leaves at 10，答案为 B 选项。\nSection 3\n讨论。\nSection 4\n讲座。",
    3: "© Rath A\n难度系数: ***\n题号 定位词 文中对应点 题目解析\nQuestions 1~6\n"
       "3. 题干: The railway was built in 1863. 定位第二段，答案为 TRUE。",
    4: "Reading Passage 2\nQuestions 14~19\n15. 题干词 tunnel 定位第五段，答案为 tunnels。",
    5: "Task 1\n考生作文点评。\n\nTask 2\n本文立场清楚。",
    6: "Part 1\nWhere do you live?\n\nPart 2\nDescribe a place.\n\nPart 3\nDiscussion.",
    7: "Part 1\n场景介绍\n第二套听力。\n\n5. 原文 the fee is 35 pounds，答案为 35。",
    8: "Part 2\n导游介绍。\nReading Passage 1\n第二套阅读。",
}


def _records():
    return to_records(segment(PAGES), [17], "17.pdf")


def test_segment_follows_listening_reading_writing_speaking_order():
    ids = [r.section_id for r in _records()]
    assert ids == [
        "cam17-t1-listening-p1", "cam17-t1-listening-p2", "cam17-t1-listening-p3", "cam17-t1-listening-p4",
        "cam17-t1-reading-p1", "cam17-t1-reading-p2",
        "cam17-t1-writing-p1", "cam17-t1-writing-p2",
        "cam17-t1-speaking-p1", "cam17-t1-speaking-p2", "cam17-t1-speaking-p3",
        "cam17-t2-listening-p1",  # "Part 1" after Speaking starts the next test
        "cam17-t2-listening-p2", "cam17-t2-reading-p1",
    ]
    reading2 = next(r for r in _records() if r.section_id == "cam17-t1-reading-p2")
    assert reading2.questions == [[14, 19]] and reading2.pages == [5]


def test_find_explanation_skips_the_answer_list_and_matches_number_or_answer():
    recs = _records()
    sec, para = find_explanation(recs, 17, "1", "listening", 2, ["dogs"])
    assert sec.part == 1 and para.startswith("2. 题干词 dog")
    sec, para = find_explanation(recs, 17, "1", "reading", 15, ["tunnels"])
    assert sec.section_id == "cam17-t1-reading-p2"
    assert find_explanation(recs, 17, "1", "listening", 9, ["wall"]) is None


def test_multi_book_file_is_split_and_general_training_relabelled():
    acad = "Section 1\n听力\nSection 2\n听力\nReading Passage 1\n阅读\nTask 1\n写作"
    gt = "Section 1\nGT 阅读\nSection 2\nGT 阅读\nTask 1\n书信"
    book = [acad] * 4 + [gt] * 2
    pages = dict(enumerate(["前言\nTask 2\n序"] + book + book))  # two books in one file
    recs = to_records(segment(pages), [4, 5], "4-5.pdf")
    tests = {(r.book, r.test) for r in recs}
    assert tests == {(b, t) for b in (4, 5) for t in ("1", "2", "3", "4", "A", "B")}
    assert {r.skill for r in recs if r.test == "A"} == {"reading", "writing"}


def test_explanation_books_from_file_names():
    assert explanation_books("4-5-6.pdf") == [4, 5, 6]
    assert explanation_books("17.pdf") == [17]
    assert explanation_books("剑19解析（A类）.pdf①.pdf") == [19]
    assert explanation_books("剑桥雅思真题精讲20学术类.pdf") == [20]
