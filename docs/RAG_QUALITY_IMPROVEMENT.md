# RAG Quality Improvement Guide

This document outlines plans and ideas for improving the quality of the Retrieval Augmented Generation (RAG) pipeline in the light-chat character AI chatbot — covering embedding, indexing, retrieval, and filtering.

## Overview

The current RAG pipeline in `conversation_manager.py`:

1. Receives the user's message as a query.
2. Looks up metadata keyword matches from a JSON keyfile (`_get_key_matches`).
3. Builds ChromaDB `$and`/`$or` filter clauses from the matches (`build_where_filters`).
4. Queries two collections — the main context collection and the `_mes` (message examples) collection — using `similarity_search_with_score`.
5. Formats retrieved chunks as `Relevant context:` in the prompt.

Each step has opportunities for improvement.

---

## 1. Chunking Strategy

### Current State

Text is split using `RecursiveCharacterTextSplitter` (or similar) with configurable `CHUNK_SIZE` (default 2048) and `CHUNK_OVERLAP` (default 1024) character counts.

### Problems

- Character-based splitting ignores sentence boundaries, producing chunks that start or end mid-sentence.
- A 2048-character chunk is quite large; the model may receive irrelevant context from the tail of a chunk that mostly contains useful information.
- Uniform chunk size means short, standalone facts and long narrative paragraphs receive the same treatment.

### Improvement Plans

#### 1.1 Sentence-Aware Splitting

Replace character-count splitting with sentence-boundary-aware splitting. Use `nltk.sent_tokenize` or `spacy` sentence segmentation to ensure chunks always start and end at sentence boundaries. Wrap this in the existing `RecursiveCharacterTextSplitter` as a custom splitter or use LangChain's `NLTKTextSplitter`.

#### 1.2 Semantic Chunking

Use semantic similarity between consecutive sentences to determine chunk boundaries — split when cosine similarity between adjacent sentence embeddings drops below a threshold. This produces thematically coherent chunks that are more aligned with how queries are formed.

Reference: LangChain's `SemanticChunker` (available in `langchain-experimental`).

#### 1.3 Smaller Target Chunk Size

Reduce default `CHUNK_SIZE` to approximately 512 characters (roughly 100–150 tokens). Smaller chunks:

- Improve retrieval precision — the returned chunk is more likely to be directly relevant.
- Reduce noise injected into the prompt.
- Allow more chunks to be returned within the same token budget.

Increase `RAG_K` to compensate for smaller individual chunks.

#### 1.4 Hierarchical Chunks (Parent-Child Indexing)

Index both small chunks (for precise retrieval) and larger parent chunks (for richer context). Store the large chunks separately. On retrieval:

1. Find relevant small chunks via similarity search.
2. Return the corresponding large parent chunk to the model.

This is supported natively in LangChain via `ParentDocumentRetriever`.

---

## 2. Embedding Model

### Current State

The system uses `HuggingFaceEmbeddings` with the default model (`all-MiniLM-L6-v2`), a 384-dimension model designed for general-purpose semantic similarity.

### Problems

- General-purpose models may not perform optimally for character dialogue or game lore domains.
- 384 dimensions may limit nuanced similarity matching for long, complex character descriptions.
- Embeddings are not normalised by default (`normalize_embeddings: False`), which can affect cosine similarity accuracy.

### Improvement Plans

#### 2.1 Enable Embedding Normalisation

Set `normalize_embeddings: True` in `encode_kwargs`. Normalised embeddings produce more consistent cosine similarity scores, which is what ChromaDB uses internally.

#### 2.2 Upgrade to a Larger Embedding Model

Consider switching to a more capable embedding model:

| Model | Dimensions | Notes |
|-------|------------|-------|
| `all-MiniLM-L6-v2` | 384 | Current default, fast, small |
| `all-mpnet-base-v2` | 768 | Better quality, moderate size |
| `BAAI/bge-base-en-v1.5` | 768 | Strong general retrieval performance |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality, larger VRAM requirement |

Make the embedding model name a configurable parameter in `appconf.json` (e.g., `EMBEDDING_MODEL`).

#### 2.3 Instruction-Prefixed Embeddings

Some models (like `BAAI/bge` series) support instruction prefixes that improve retrieval by signalling whether the embedding is for a query or a document:

- At index time: `"Represent this document: <chunk_text>"`
- At query time: `"Represent this question for searching relevant passages: <query>"`

The `HuggingFaceEmbeddings` class supports `query_instruction` and `embed_instruction` parameters.

---

## 3. Metadata Filtering

### Current State

`extract_key_matches` checks whether any metadata keyword appears in the user's raw query string. Matches are used to build `$and` or `$or` ChromaDB `where` filters. This is a simple exact substring match.

### Problems

- Exact substring matching misses queries that use synonyms or paraphrases (e.g., "the space station" matches "station" but not "Citadel").
- Very short keywords (e.g., "God", "dance", "minute") may match queries where they are incidental.
- The `$and` filter may be too restrictive when multiple keywords match — if a chunk only contains one of the matched keywords, it is excluded.
- No weighting: a rare, highly specific term (e.g., "Quantum Bio-Reconstruction Machine") is treated identically to a common term (e.g., "station").

### Improvement Plans

#### 3.1 Fuzzy / Normalised Matching

Normalise both the query and keyword values before matching (lowercase, strip punctuation, remove common stopwords). This catches more matches without introducing false positives.

#### 3.2 Prefer `$or` Over `$and` for Initial Retrieval

Use `$or` filtering as the primary strategy to maximise recall. Use `$and` only as a secondary high-precision attempt, with fallback to `$or` and then unfiltered:

```
Try: $and filter → if results < threshold, try: $or filter → if no results, try: unfiltered
```

#### 3.3 Category-Based Filter Weights

If metadata entries are extended with `category` (see `RAG_DOCUMENT_QUALITY.md`), apply different matching strategies per category:

- `character` and `location` entries: high-weight, match even in short queries.
- `date` entries: only match if date pattern appears in query.
- Generic concept entries: require longer query overlap before filtering.

#### 3.4 Query Entity Extraction

Before searching, apply lightweight Named Entity Recognition (NER) to the user's query to extract proper nouns and specific terms. Use these extracted entities to drive metadata filtering instead of relying on substring matching across the full keyword list.

Python libraries: `spacy` (small model) or even a simple regex-based proper noun extractor.

---

## 4. Query Expansion and Reformulation

### Current State

The user's raw message is passed directly as the RAG query. No expansion or reformulation is applied.

### Problems

- Short queries (e.g., "Who are you?") may not retrieve relevant context.
- Colloquial phrasing may not match formal document vocabulary.
- Single-query retrieval can miss relevant chunks that use different phrasing.

### Improvement Plans

#### 4.1 Multi-Query Retrieval

Generate 2–3 alternative phrasings of the user's query using a small LLM call or a template-based approach. Retrieve results for all variants and merge/deduplicate. This increases recall without sacrificing precision.

Example: "Who made you?" → also search "SHODAN origin", "SHODAN creation", "TriOptimum SHODAN".

#### 4.2 Contextual Query Enrichment

Prepend the character name and topic context to the query before retrieval:

```python
enriched_query = f"{self.character_name} {user_message}"
```

This helps the embedding model orient the query toward the character's domain.

#### 4.3 History-Aware Queries

Use the most recent conversation turn(s) to enrich the query. If the last exchange was about Citadel Station, the current query "what happened there?" should search with Citadel Station context. Concatenate recent conversation context to the query:

```python
context_query = f"{recent_history_summary} {user_message}"
```

---

## 5. Retrieval Re-Ranking

### Current State

Retrieved chunks are returned in cosine similarity order as computed by ChromaDB. No re-ranking is applied.

### Problems

- Cosine similarity measures semantic relatedness but not answer quality or specificity.
- The top-k chunks may include repetitive or tangentially related content.
- ChromaDB similarity scores are not calibrated — a score of 0.8 does not mean the same thing across different queries.

### Improvement Plans

#### 5.1 Cross-Encoder Re-Ranking

After retrieving an initial candidate set (e.g., top 15 chunks), re-rank using a cross-encoder model that takes (query, chunk) pairs and predicts relevance:

```
Initial retrieval: k=15 candidates
Cross-encoder score each (query, chunk) pair
Return top k=5 or k=7 after re-ranking
```

Recommended models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, small).

LangChain supports this via `CrossEncoderReranker` with `FlashrankRerank` or `CohereRerank`.

#### 5.2 Score Thresholding

Filter out chunks below a minimum relevance score. If the best match has a score below a threshold (e.g., 0.4 for normalised embeddings), return no context rather than irrelevant context:

```python
docs = [(doc, score) for doc, score in docs if score >= MIN_SCORE_THRESHOLD]
```

This prevents low-quality context from being injected into the prompt.

#### 5.3 Maximal Marginal Relevance (MMR)

Use MMR retrieval instead of pure similarity search to reduce redundancy in the retrieved set. MMR selects chunks that are both relevant to the query and diverse from each other:

```python
db.max_marginal_relevance_search(query, k=7, fetch_k=20)
```

ChromaDB and LangChain's `Chroma` class already support MMR search.

---

## 6. Separate Retrieval for Context vs Examples

### Current State

The same metadata filters and the same `k` parameter are used for both the main context collection and the `_mes` (message examples) collection.

### Problems

- Message examples have a very different distribution to context chunks — they are short dialogue snippets rather than lore prose.
- Filtering message examples with the same metadata filters as context documents may over-restrict example retrieval.
- `k` for message examples should probably be lower (fewer examples needed) than for context.

### Improvement Plans

#### 6.1 Separate `k` Values

Add a configurable `RAG_K_MES` parameter distinct from `RAG_K` for controlling how many message example chunks are retrieved.

#### 6.2 Lighter Filtering for Examples

For the `_mes` collection, fall back to unfiltered or `$or`-only retrieval more aggressively. The goal is to find stylistically similar dialogue, not factually matching information.

#### 6.3 Deduplicate Across Collections

After retrieving from both collections, deduplicate chunks that appear in both (e.g., if a message example happens to duplicate content in the context document).

---

## 7. Contextual Compression

### Current State

Full retrieved chunks are injected into the prompt verbatim. A 2048-character chunk may contain only 1–2 sentences that are actually relevant to the current query.

### Problems

- The model receives a lot of context noise, diluting the signal.
- Large context chunks consume token budget that could be used for conversation history.

### Improvement Plans

#### 7.1 LLM-Based Contextual Compression

Use a small, fast LLM call to extract only the relevant sentences from each retrieved chunk:

```
"Given the question: {query}
Extract only the sentences from the following passage that are directly relevant:
{chunk}"
```

LangChain provides `ContextualCompressionRetriever` with `LLMChainExtractor` for this purpose.

#### 7.2 Sentence-Level Re-Ranking

After retrieval, split each chunk into sentences and score each sentence against the query using the embedding model. Return only the top-n sentences per chunk. This is computationally lightweight compared to LLM-based compression.

---

## 8. Collection Management

### Current State

Collections are created once and overwritten manually. There is no versioning or incremental update mechanism.

### Improvement Plans

#### 8.1 Incremental Updates

Instead of rebuilding the entire collection when source documents change, compute a diff between old and new documents and only re-embed changed chunks.

#### 8.2 Collection Versioning

Name collections with a version suffix (e.g., `shodan_v2`, `shodan_v3`) and keep previous versions available for rollback. Update `RAG_COLLECTION` in `appconf.json` to switch versions.

#### 8.3 Offline Quality Metrics

After building a collection, run a set of predefined test queries and record:

- Average retrieval score.
- Hit rate (does the expected chunk appear in the top-k results?).
- Diversity of retrieved chunks.

Store these metrics alongside each collection version to track quality over time.

---

## Priority Recommendations

| Priority | Improvement | Expected Impact |
|----------|-------------|-----------------|
| High | Enable `normalize_embeddings: True` | Immediate accuracy improvement |
| High | Use MMR for retrieval | Reduces redundant context |
| High | Score thresholding | Prevents irrelevant context injection |
| Medium | Sentence-aware chunking | Better chunk boundary quality |
| Medium | Separate k for context vs examples | More balanced prompt content |
| Medium | Query entity enrichment | Improves short-query retrieval |
| Low | Cross-encoder re-ranking | Higher precision, more compute |
| Low | Multi-query retrieval | Higher recall, more API calls |

---

## See Also

- [RAG_DOCUMENT_QUALITY.md](RAG_DOCUMENT_QUALITY.md) — Improving source document quality
- [CONVERSATION_QUALITY.md](CONVERSATION_QUALITY.md) — Improving overall conversation quality
- [RAG_SCRIPTS_GUIDE.md](RAG_SCRIPTS_GUIDE.md) — How to use the RAG management scripts
- `core/conversation_manager.py` — RAG retrieval implementation
- `core/collection_helper.py` — Collection management utilities
