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
            "Standard customers have a 30-day return window"
            "If a Standard customer's order gets delayed, they get a $20 credit as compensation"
            "The return window starts from the day of delivery"
        ),
        "metadata": {"category": "return_policy", "tier": "Standard"},
    },
    {
        "id": "vip_return_policy",
        "content": (
            "VIP customers have a 60-day return window"
            "If a VIP customer's order gets delayed or receive any damage, they get a full refund as compensation - no questions asked"
            "The return window starts from the day of delivery"
        ),
        "metadata": {"category": "return_policy", "tier": "VIP"},
    },
]

def _tokenize(text: str) -> List[str]:
    """Split sentence into tokens"""
    text = text.lower()
    text = text = re.sub(r"[^a-z0-9\s]", " ", text)
    

# try:
#     from langchain_community.vectorstores import FAISS
#     from langchain_google_genai import GoogleGenerativeAIEmbeddings
#     from langchain_core.documents import Document

#     def _build_faiss_index():
#         docs = [
#             Document(page_content=p["content"], metadata=p["metadata"])
#             for p in POLICY_DOCUMENTS
#         ]
#         # Use a currently supported Google Gemini embedding model
#         # older model names like "models/embedding-001" are no longer valid
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-3-large")
#         return FAISS.from_documents(docs, embedding=embeddings)

#     _FAISS_INDEX = _build_faiss_index()

#     def retrieve_policy(query: str, top_k: int = 2) -> dict:
#         # perform similarity search over the pre-built FAISS index
#         results = _FAISS_INDEX.similarity_search(query, k=top_k)
#         context_str = "\\n\\n".join(
#             f"[Policy Chunk {i+1}]\\n{r.page_content}"
#             for i, r in enumerate(results)
#         )
#         return {
#             "query": query,
#             "results": [{"content": r.page_content, "metadata": r.metadata} for r in results],
#             "context": context_str,
#         }

# except ImportError:
#     pass  # Fallback to keyword retriever above