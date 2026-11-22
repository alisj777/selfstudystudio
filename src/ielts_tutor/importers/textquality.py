"""Score and repair OCR text from scanned exam books.

`Vocab.ratio(text)` is the share of word tokens found in an English word list. Clean
typeset English scores about 0.95; OCR of handwriting or broken text layers scores well
below 0.8, which is how importers decide what is safe to index.
"""

from __future__ import annotations

import re
from pathlib import Path

SYSTEM_WORDLIST = Path("/usr/share/dict/words")
_TOKEN = re.compile(r"[A-Za-z]+")
_SUFFIXES = ("'s", "s", "es", "ed", "d", "ing", "ly", "er", "ers", "est", "ies", "ied", "ment", "ness")
# Common modern words missing from older dictionaries such as web2.
_EXTRA = """
internet online email website websites smartphone smartphones app apps computer computers technology
television tv online percent percentage okay ok ielts download upload software laptop mobile video
""".split()


class Vocab:
    def __init__(self, words: set[str]):
        self.words = words

    @classmethod
    def load(cls, path: Path = SYSTEM_WORDLIST, extra_text: str = "") -> Vocab:
        words = set(_EXTRA)
        if path.exists():
            words |= {w.strip().lower() for w in path.read_text(encoding="utf-8", errors="ignore").split()}
        words |= {w.lower() for w in _TOKEN.findall(extra_text) if len(w) > 2}
        return cls(words)

    def known(self, word: str) -> bool:
        w = word.lower()
        if len(w) == 1:
            return w in ("a", "i")
        if w in self.words:
            return True
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                stem = w[: -len(suf)]
                if stem in self.words or stem + "e" in self.words or (suf in ("ies", "ied") and stem + "y" in self.words):
                    return True
                if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in self.words:  # running -> run
                    return True
        return False

    def ratio(self, text: str) -> float:
        tokens = _TOKEN.findall(text)
        if not tokens:
            return 0.0
        return sum(self.known(t) for t in tokens) / len(tokens)

    def respace(self, line: str) -> str:
        """Re-join words that OCR split into letters ("th e gro und" -> "the ground").

        Only applied when many tokens are short unknown fragments; merges the longest run
        of up to six adjacent tokens that forms a known word.
        """
        tokens = line.split()
        if len(tokens) < 4:
            return line
        fragments = sum(1 for t in tokens if len(_core(t)) <= 3 and not self.known(_core(t) or "x"))
        if fragments / len(tokens) < 0.2:
            return line
        out, i = [], 0
        while i < len(tokens):
            for j in range(min(len(tokens), i + 6), i + 1, -1):
                # A whole known word after the first token stays separate ("fo r ages" -> "for ages").
                if any(len(_core(t)) > 2 and self.known(_core(t)) for t in tokens[i + 1 : j]):
                    continue
                cand = "".join(tokens[i:j])
                core = _core(cand)
                if core and _TOKEN.fullmatch(core) and self.known(core):
                    out.append(cand)
                    i = j
                    break
            else:
                out.append(tokens[i])
                i += 1
        return " ".join(out)


def _core(token: str) -> str:
    return token.strip(".,;:!?\"'()[]-–—")
