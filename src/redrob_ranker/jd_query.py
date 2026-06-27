"""
The free-text representation of the JD used for semantic (embedding/TF-IDF)
matching against candidate documents.

This is deliberately written the way the JD's own "how to read between the
lines" section describes the *ideal candidate* -- not a keyword dump of the
"skills inventory" section. The JD explicitly says the trap is rewarding
keyword overlap; we don't want our own retrieval query to recreate that trap
by being a list of nouns. So this text reads like a strong candidate's
narrative summary, which is exactly the kind of text candidate profiles
contain in their `profile.summary` and `career_history[].description` fields.
"""

JD_QUERY_TEXT = """
Senior AI Engineer, founding engineering team at an AI-native talent
intelligence platform. Owns the ranking, retrieval, and matching systems that
decide what recruiters see when they search for candidates. Has shipped an
end-to-end ranking, search, or recommendation system to real users at
meaningful scale, at a product company (not a pure-services or consulting
shop). Deep hands-on production experience with embeddings-based retrieval --
sentence-transformers, OpenAI embeddings, BGE, or E5 -- including embedding
drift, index refresh, and retrieval-quality regression in production. Operated
a vector database or hybrid search stack such as Pinecone, Weaviate, Qdrant,
Milvus, OpenSearch, Elasticsearch, or FAISS in production, combining dense and
lexical (BM25) retrieval. Strong Python and systems engineering, not just
notebook prototyping. Has designed rigorous evaluation frameworks for ranking
systems -- NDCG, MRR, MAP, offline-to-online correlation, A/B testing -- and
can reason about retrieval-quality tradeoffs from first principles, not just
from following tutorials. Understood retrieval and ranking before it became
fashionable; pre-LLM-era production ML experience, not only recent
LangChain-calls-an-LLM projects. Comfortable with both deep technical depth in
modern ML systems and a scrappy, ship-fast product-engineering mindset.
Bonus: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models,
distributed systems or large-scale inference optimization, HR-tech or
marketplace product background, open-source contributions or public writing
about systems they built.
""".strip()
