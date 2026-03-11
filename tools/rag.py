from typing import List, Tuple
from dotenv import load_dotenv
import re
import math
from collections import Counter

load_dotenv()

POLICY_DOCUMENTS = [
    {
        "id": "standard_return_policy",
        "content": (
            "Standard customers have a 30-day return window. "
            "If a Standard customer's order gets delayed, they get a $20 credit as compensation. "
            "The return window starts from the day of delivery."
        ),
        "metadata": {"category": "return_policy", "tier": "Standard"},
    },
    {
        "id": "vip_return_policy",
        "content": (
            "VIP customers have a 60-day return window. "
            "If a VIP customer's order gets delayed or receive any damage, they get a full refund as compensation - no questions asked. "
            "The return window starts from the day of delivery."
        ),
        "metadata": {"category": "return_policy", "tier": "VIP"},
    },
]

def _tokenize(text: str) -> List[str]:
    """Split sentence into tokens"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

def _compute_tf(tokens: List[str]) -> dict:
    """Compute term frequency for a list of tokens"""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counts.items()}


class TFIDFRetriever:
    """This is an implmentation fo a simple TF-IDF based retriever."""
    def __init__(self, documents: List[dict]):
        self.documents = documents
        self.doc_tokens : List[List[str]] = []
        self.doc_tfs : List[dict] = []
        self.idf : dict = {}
        self._build_index()

    def _build_index(self):
        N = len(self.documents)

        for doc in self.documents:
            tokens = _tokenize(doc["content"])
            self.doc_tokens.append(tokens)
            self.doc_tfs.append(_compute_tf(tokens))
        
        df: dict = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1

        self.idf = {
            term: math.log((N + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

    def _tfidf_score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_tf = self.doc_tfs[doc_idx]
        score = 0.0

        for term in query_tokens:
            if term in doc_tf and term in self.idf:
                score += doc_tf[term] * self.idf[term]

        return score / (len(query_tokens) + 1)
    
    def retrieve(self, query: str, top_k: int = 2) -> List[dict]:
        query_tokens = _tokenize(query)
        
        scored: List[Tuple[float, int]] = []
        for i in range(len(self.documents)):
            score = self._tfidf_score(query_tokens, i)
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, idx in top:
            if score > 0:
                results.append({
                    "title": self.documents[idx]["id"],
                    "text": self.documents[idx]["content"],
                    "relevance_score": round(score, 4),
                })

        return results

_retriever = TFIDFRetriever(POLICY_DOCUMENTS)

def retrieve_policy(query: str, top_k: int = 2) -> dict:
    results = _retriever.retrieve(query, top_k=top_k)

    if not results:
        return {
            "success": False,
            "error": "No relevant policy found for that query.",
            "context": "",
        }

    context_parts = []
    for r in results:
        context_parts.append(f"[{r['title']}]\n{r['text']}")

    return {
        "success": True,
        "results": results,
        "context": "\n\n".join(context_parts),
    }