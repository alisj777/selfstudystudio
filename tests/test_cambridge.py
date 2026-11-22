from pathlib import Path

from ielts_tutor.importers.cambridge import audio_manifest, convert, parse_book, writing_type
from ielts_tutor.importers.textquality import Vocab
from ielts_tutor.rag.build import rubric_docs
from ielts_tutor.rag.schema import ScoredSample

VOCAB = Vocab({w for w in """
the a of and to in is that people children school should be for this candidate answer clear ideas
range vocabulary grammar errors are some good used with there it well about at book room would like
reserve how many guests will come so i can how much time do you have glad hear we organised paragraphs
describe someone who teaches you what they teach why explain whether learned lot from them
city traffic problem cars many more roads busy every day not always repeated adequate but has
""".split()})

FRONT = """Test 1
LISTENING
SECTION 1 Questions 1-10
READING
READING PASSAGE 1
The history of tea is long and the history of tea is interesting to many people in the world.
Questions 1-13
WRITING
WRITING TASK 1
You should spend about 20 minutes on this task.
The table below shows the number of visitors to three museums in 2010 and 2020.
Summarise the information by selecting and reporting the main features.
Write at least 150 words.
WRITING TASK 2
You should spend about 40 minutes on this task.
Write about the following topic:
Some people think children should start school at a very young age.
To what extent do you agree or disagree?
Writeatleast250words.
SPEAKING
PART 1
The examiner asks the candidate about him/herself.
Hometown
• Where is your hometown? • What do you like about it?
PART 2
You will have to talk about the topic for one to two minutes.
PART 3
Discussion topics:
Teachers
What makes a good teacher? Why do people choose teaching as a career?
Describe someone who teaches you.
You should say:
who they are
what they teach
and explain whether you learned a lot from them.
"""

BACK = """Audioscripts
TEST 1
SECTION 1
MAN: I would like to book a room. WOMAN: How many guests will come so I can reserve it?
SECTION 2
WOMAN: We are glad to hear that the city traffic problem is busy every day with many more cars.
Sample answers for Writing tasks
TEST 1, WRITING TASK 1
This is an answer written by a candidate who achieved a Band 6.5 score.
The table shows how many people visited three museums. In 2010 the science museum had the most
visitors and the art museum had the fewest. By 2020 the numbers had grown for all three of them.
Overall the science museum stayed the most popular and the history museum grew the fastest.
The art museum doubled its visitors while the science museum grew by a third over the period.
In conclusion all three museums became more popular and the gap between them became smaller.
Here is the examiner's comment:
The candidate covers the key features and the ideas are clear. The range of vocabulary is good
and grammar errors are few. Some paragraphs are well organised with a clear overview of trends.
TEST 1, WRITING TASK 2
This is an answer written by a candidate who achieved a Band 5 score. Here is the examiner's
comment:
The candidate has some good ideas and there are some clear paragraphs.
The range of vocabulary is adequate but there are some errors in grammar.
Ideas are not always well organised with this answer and some are repeated.
Chi1dr- en sh-ould sta- rt sc~ool at a yo- ung a~e bec- ause t~ey le- arn qu~ckly.
T~e pa- rents al~o ca~ wo- rk wh~le t~e ch~ldren a- re at sc~ool ev- ery d~y.
I~ my op~n~on th~s ~s a go- od ide~ fo- r ev- ery fa~ily ~n th- e ci~y.
"""


def _parsed():
    return parse_book([FRONT, BACK], book=99)


def test_parse_book_finds_each_part_of_a_test():
    items = {(i.kind, i.test, i.n) for i in _parsed().items}
    assert ("reading", 1, 1) in items
    assert ("writing", 1, 1) in items and ("writing", 1, 2) in items
    assert {("speaking", 1, n) for n in (1, 2, 3)} <= items
    assert ("transcript", 1, 1) in items and ("transcript", 1, 2) in items


def test_convert_builds_chunks_and_moves_cue_card_to_part2(tmp_path):
    res = convert(_parsed(), VOCAB, {(99, 1, 1): tmp_path / "t1s1.mp3"}, Path("book.pdf"))
    ch = {c.chunk_id: c for c in res.chunks}
    assert ch["cam99-t1-w1"].text.startswith("The table below shows")
    assert "minutes on this task" not in ch["cam99-t1-w1"].text
    assert ch["cam99-t1-w1"].question_type == "qt.writing.t1.table"
    assert ch["cam99-t1-w2"].question_type == "qt.writing.t2.opinion"
    assert "Describe someone who teaches you" in ch["cam99-t1-s2"].text
    assert "Describe" not in ch["cam99-t1-s3"].text
    assert ch["cam99-t1-l1"].media_uri.endswith("t1s1.mp3")
    assert ch["cam99-t1-l1"].source_type == "transcript"


def test_samples_split_comment_and_essay_in_both_layouts():
    res = convert(_parsed(), VOCAB, {}, Path("book.pdf"))
    by_task = {s.task: s for s in res.samples}
    new = by_task["task1"]  # essay first, then "Here is the examiner's comment:"
    assert new.scores == {"overall": 6.5} and new.rater == "official"
    assert new.response_text.startswith("The table shows")
    assert new.comments.startswith("The candidate covers")
    assert new.prompt_text.startswith("The table below shows")
    old = by_task["task2"]  # comment first, then the scanned handwritten essay
    assert old.scores == {"overall": 5.0}
    assert old.comments.endswith("some are repeated.")
    assert old.response_text.startswith("Chi1dr")
    assert old.quality < 0.9


def test_writing_type_rules():
    assert writing_type(1, "The maps below show the village in 1990 and now.", "academic") == "qt.writing.t1.map"
    assert writing_type(1, "The table and pie chart give information about ...", "academic") == "qt.writing.t1.mixed"
    assert writing_type(1, "Write a letter to your landlord.", "general") is None
    assert writing_type(2, "Discuss both these views and give your own opinion.", "academic") == "qt.writing.t2.discussion"
    assert writing_type(2, "Do the advantages outweigh the disadvantages?", "academic") == "qt.writing.t2.adv_disadv"
    assert writing_type(2, "What are the causes of this problem? What solutions can you suggest?", "academic") == (
        "qt.writing.t2.problem_solution"
    )


def test_respace_rejoins_split_words_but_keeps_real_words():
    v = Vocab({"the", "ground", "for", "ages", "that", "lie", "seeds", "in"})
    assert v.respace("seeds th a t lie in th e ground fo r ages") == "seeds that lie in the ground for ages"
    assert v.respace("seeds that lie in the ground") == "seeds that lie in the ground"


def test_audio_manifest_reads_varied_file_names(tmp_path):
    for rel in ["剑4/C4T1S2.mp3", "剑11/IELTS11_Test3_Section4.mp3", "剑雅17音频/ELT_IELTS17_t2_audio1.mp3",
                "剑12/IELTS 12 Test 5_S1.mp3", "剑18听力音频/18-1-3.mp3", "剑18听力音频/Test 1 Track 04.mp3"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    found = audio_manifest(tmp_path)
    assert set(found) == {(4, 1, 2), (11, 3, 4), (17, 2, 1), (12, 5, 1), (18, 1, 3)}


def test_unreadable_essay_keeps_only_the_examiner_comment(kg):
    noisy = ScoredSample(sample_id="cam-x", skill="writing", task="task2", response_text="t~e ch~ld",
                         scores={"overall": 5.0}, rater="official", comments="Ideas are unclear.", quality=0.4)
    clean = noisy.model_copy(update={"sample_id": "cam-y", "response_text": "The child", "quality": 0.97})
    kinds = {(d.id, d.meta["kind"]) for d in rubric_docs(kg, [noisy, clean]) if d.id.startswith("cam-")}
    assert kinds == {("cam-x#comment", "examiner_comment"), ("cam-y", "sample"), ("cam-y#comment", "examiner_comment")}
