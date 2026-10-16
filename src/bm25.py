"""
Keyword search (BM25) - the "sparse" half of hybrid retrieval.

Dense embeddings are good at meaning ("stop browsers blocking my API" ->
CORS), but weak at exact identifiers like `jsonable_encoder` or
`status_code=201`, which developers type all the time. BM25 scores
documents by exact term overlap, so combining the two covers both kinds
of query.

The corpus is ~500 chunks, so the index is built in memory at startup in
a few milliseconds - no extra service or dependency needed.
"""

import math
import re
from collections import Counter, defaultdict

WORD_RE = re.compile(r"[a-z0-9]+")
IDENT_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "should", "so", "that", "the", "their", "then", "there", "this",
    "to", "use", "using", "was", "what", "when", "where", "which", "why", "will",
    "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """
    Lowercased words, plus whole snake_case identifiers kept intact so
    "allow_origins" matches both as one token and as "allow" + "origins".
    """
    text = text.lower()
    words = [w for w in WORD_RE.findall(text) if w not in STOPWORDS]
    return words + IDENT_RE.findall(text)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)

        for i, doc in enumerate(documents):
            freqs = Counter(tokenize(doc))
            self.doc_freqs.append(freqs)
            self.doc_lens.append(sum(freqs.values()))
            for term in freqs:
                self.postings[term].append(i)

        n = len(documents)
        self.avg_len = sum(self.doc_lens) / n if n else 0
        self.idf = {
            term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Returns up to k (document index, score) pairs, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.doc_freqs[i][term]
                norm = 1 - self.b + self.b * self.doc_lens[i] / self.avg_len
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
