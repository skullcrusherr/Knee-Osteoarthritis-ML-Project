# chatbot.py
import os, re, glob
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAFE_MEDICAL_DISCLAIMER = (
    "I’m not a clinician. I can share general info and cite your local notes, "
    "but I can’t diagnose or provide treatment. For medical advice, consult a professional."
)

@dataclass
class DocChunk:
    doc_id: str
    section: str
    text: str

@dataclass
class Retrieval:
    answer: str
    citations: List[Tuple[str, str]]  # [(filename, section)]
    debug: Optional[List[Tuple[str, float]]] = None  # [(doc_id:section, sim)]

class KBIndex:
    def __init__(self, knowledge_dir: str = "knowledge", chunk_size: int = 800, overlap: int = 120):
        self.knowledge_dir = knowledge_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.chunks: List[DocChunk] = []

    def _read_docs(self) -> List[Tuple[str, str]]:
        files = sorted(glob.glob(os.path.join(self.knowledge_dir, "**", "*.*"), recursive=True))
        out = []
        for f in files:
            if os.path.splitext(f)[1].lower() in [".md", ".txt"]:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    out.append((f, fh.read()))
        return out

    def _split_markdown(self, fname: str, text: str) -> List[DocChunk]:
        # heading-aware split + fixed-size fallback
        sections = re.split(r"(?m)^\s*#+\s+", text)
        heads = re.findall(r"(?m)^\s*#+\s+(.*)$", text)
        chunks: List[DocChunk] = []

        if len(sections) <= 1:
            chunks.extend(self._fixed_chunks(fname, "Chunk", text))
            return chunks

        if sections[0].strip():
            chunks.extend(self._fixed_chunks(fname, "Preface", sections[0]))

        for head, body in zip(heads, sections[1:]):
            chunks.extend(self._fixed_chunks(fname, head.strip(), body))
        return chunks

    def _fixed_chunks(self, fname: str, section: str, text: str) -> List[DocChunk]:
        # normalize whitespace for better TF-IDF
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        cs, ov = self.chunk_size, self.overlap
        out = []
        start = 0
        while start < len(text):
            out.append(DocChunk(
                doc_id=os.path.basename(fname),
                section=section,
                text=text[start:start+cs]
            ))
            start += max(1, cs - ov)
        return out

    def build(self):
        self.chunks.clear()
        docs = self._read_docs()
        for f, txt in docs:
            self.chunks.extend(self._split_markdown(f, txt))

        if not self.chunks:
            raise RuntimeError("No knowledge files found. Add .md or .txt files to the 'knowledge/' folder.")

        corpus = [c.text for c in self.chunks]
        # Slightly richer TF-IDF; robust to short Qs
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 3),            # up to 3-grams helps Q/A phrasing
            sublinear_tf=True,
            max_features=60000,
            min_df=1,
            max_df=0.98
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    @staticmethod
    def _normalize_query(q: str) -> str:
        q = q.strip()
        q = re.sub(r"^\s*(q:)\s*", "", q, flags=re.I)  # remove leading "Q:"
        return q

    def query(self, q: str, k: int = 3, min_sim: float = 0.12) -> Retrieval:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Index not built. Call build() first.")

        q = self._normalize_query(q)
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        top = [(self.chunks[i], float(sims[i])) for i in top_idx]

        # refuse if top similarity too low
        if not top or top[0][1] < min_sim:
            return Retrieval(
                answer="I don’t have a confident answer in the local knowledge base.",
                citations=[],
                debug=[(f"{self.chunks[i].doc_id}:{self.chunks[i].section}", float(sims[i])) for i in top_idx]
            )

        # compose concise answer (use just the best chunk to avoid rambling)
        best_chunk, best_sim = top[0]
        synthesis = best_chunk.text.strip()

        # Trim to ~700 chars nicely
        if len(synthesis) > 700:
            synthesis = synthesis[:700].rsplit(" ", 1)[0] + "…"

        citations = [(best_chunk.doc_id, best_chunk.section)]
        debug = [(f"{c.doc_id}:{c.section}", s) for c, s in top]
        return Retrieval(answer=synthesis, citations=citations, debug=debug)

def is_medical_advice(q: str) -> bool:
    triggers = [
        "treat", "treatment", "medication", "dose", "prescribe", "diagnose",
        "what should i take", "should i take", "how much", "surgery", "operate",
        "my knee xray", "my x-ray", "my x ray", "is this", "do i have"
    ]
    ql = q.lower()
    return any(t in ql for t in triggers)

def answer_query(kb: KBIndex, q: str) -> Retrieval:
    if is_medical_advice(q):
        return Retrieval(answer=SAFE_MEDICAL_DISCLAIMER, citations=[])
    return kb.query(q, k=3, min_sim=0.12)
