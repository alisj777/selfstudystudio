"""Hand-made essay features (design doc 6.1, step 2).

Cheap, explainable signals that track the four writing criteria:
  TR   length against the task minimum, relevance to the prompt
  CC   paragraphing, linking words (count and variety), sentence-to-sentence flow
  LR   lexical diversity (MTLD), word length, rare words, spelling errors
  GRA  sentence length and spread, subordinate clauses, punctuation, error density

Every feature is a number so the scorer can use them directly, and the analyzer can
quote them back to the student ("you used 4 different linking words").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from ..importers.textquality import Vocab

LINKERS = [
    "however", "moreover", "furthermore", "in addition", "additionally", "therefore", "thus", "hence",
    "consequently", "as a result", "for example", "for instance", "such as", "in conclusion", "to conclude",
    "to sum up", "overall", "on the other hand", "in contrast", "whereas", "while", "although", "even though",
    "despite", "in spite of", "nevertheless", "nonetheless", "firstly", "secondly", "finally", "lastly",
    "first of all", "in my opinion", "in my view", "similarly", "likewise", "because", "since", "due to",
    "owing to", "meanwhile", "instead", "otherwise", "indeed", "in fact", "that is", "namely", "besides",
]
SUBORDINATORS = {"because", "although", "though", "which", "who", "whom", "whose", "that", "when", "whereas",
                 "while", "if", "unless", "since", "whether", "where", "until", "once", "as"}
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


@dataclass
class EssayFeatures:
    words: int
    words_vs_min: float  # words / task minimum (150 or 250), capped at 2
    paragraphs: int
    sentences: int
    mean_sentence_len: float
    sd_sentence_len: float
    long_sentence_ratio: float  # sentences > 30 words
    mtld: float
    ttr_first_200: float
    mean_word_len: float
    long_word_ratio: float  # >= 7 letters
    unknown_word_ratio: float  # not in the dictionary: misspellings, mostly
    linker_count: int
    linker_types: int
    linkers_per_sentence: float
    subordinate_per_sentence: float
    commas_per_sentence: float
    repeated_bigram_ratio: float
    first_person_ratio: float
    prompt_overlap: float  # share of prompt content words used in the essay
    grammar_errors_per_100w: float = -1.0  # from LanguageTool when available, else -1

    def vector(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def names() -> list[str]:
        return list(EssayFeatures.__dataclass_fields__)


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010), forward and backward."""

    def one_way(seq: list[str]) -> float:
        factors, types, count = 0.0, set(), 0
        for tok in seq:
            count += 1
            types.add(tok)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0
        return len(seq) / factors if factors else float(len(seq))

    if len(tokens) < 10:
        return float(len(tokens))
    return (one_way(tokens) + one_way(tokens[::-1])) / 2


_STOP = set("""a an the and or but of to in on at for with by from as is are was were be been being it its this
that these those there their they them he she his her we our you your i my me not no do does did have has had
will would can could should may might must so such than then very more most many much some any all also""".split())


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 2}


def extract(essay: str, prompt: str | None = None, task: str = "task2", vocab: Vocab | None = None,
            grammar_errors: int | None = None) -> EssayFeatures:
    text = essay.strip()
    tokens = [w.lower() for w in _WORD.findall(text)]
    n = len(tokens) or 1
    paragraphs = [p for p in re.split(r"\n\s*\n|\n(?=\s*[A-Z])", text) if p.strip()]
    sentences = [s for s in _SENT.split(" ".join(text.split())) if _WORD.search(s)] or [text]
    lens = [len(_WORD.findall(s)) for s in sentences]
    mean_len = sum(lens) / len(lens)
    sd_len = math.sqrt(sum((x - mean_len) ** 2 for x in lens) / len(lens))
    low = " " + " ".join(tokens) + " "
    linkers = Counter(l for l in LINKERS for _ in range(low.count(f" {l} ")))
    bigrams = list(zip(tokens, tokens[1:]))
    bigram_counts = Counter(bigrams)
    repeated = sum(c for b, c in bigram_counts.items() if c > 2 and not set(b) <= _STOP) / max(len(bigrams), 1)
    vocab = vocab or Vocab.load()
    unknown = sum(1 for w in tokens if len(w) > 2 and not vocab.known(w)) / n
    p_words = content_words(prompt or "")
    overlap = len(p_words & set(tokens)) / len(p_words) if p_words else 0.0
    minimum = 150 if task == "task1" else 250
    return EssayFeatures(
        words=len(tokens),
        words_vs_min=min(len(tokens) / minimum, 2.0),
        paragraphs=len(paragraphs),
        sentences=len(sentences),
        mean_sentence_len=mean_len,
        sd_sentence_len=sd_len,
        long_sentence_ratio=sum(1 for x in lens if x > 30) / len(lens),
        mtld=_mtld(tokens),
        ttr_first_200=len(set(tokens[:200])) / max(min(len(tokens), 200), 1),
        mean_word_len=sum(len(w) for w in tokens) / n,
        long_word_ratio=sum(1 for w in tokens if len(w) >= 7) / n,
        unknown_word_ratio=unknown,
        linker_count=sum(linkers.values()),
        linker_types=len(linkers),
        linkers_per_sentence=sum(linkers.values()) / len(sentences),
        subordinate_per_sentence=sum(1 for w in tokens if w in SUBORDINATORS) / len(sentences),
        commas_per_sentence=text.count(",") / len(sentences),
        repeated_bigram_ratio=repeated,
        first_person_ratio=sum(1 for w in tokens if w in {"i", "my", "me"}) / n,
        prompt_overlap=overlap,
        grammar_errors_per_100w=-1.0 if grammar_errors is None else 100 * grammar_errors / n,
    )
