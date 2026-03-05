# Context Flow Visualizations

Last verified: 2026-03-06

## High-Level Runtime Pipeline

```text
User Input
   |
   v
Heuristic Gate (small-talk / follow-up skip)
   |
   +--> Skip RAG -> minimal context placeholder
   |
   +--> Use RAG
            |
            v
      Query enrichment (character + input)
            |
            v
      Retrieve lore + _mes chunks
            |
            v
      Optional reranking of top candidates
            |
            v
      Filter/dedupe/cap context text
            |
            v
      Dynamic allocation (non-first turns)
            |
            v
      Build prompt (mistral/custom OR template path)
            |
            v
      Stream response with safeguards
            |
            v
      Post-process + quality gate + history update
```

## Dynamic Budget Concept

```text
Total context window
  - reserved response tokens
  - system prompt tokens
  --------------------------------
  = available context budget

available context budget
  - input tokens
  --------------------------------
  = dynamic allocation pool

allocation pool split across:
  history / message examples / vector context
```

## Retrieval Fallback Ladder

```text
if metadata matches:
  try $and filter
  if no useful results -> try $or filter
  if still no useful results -> unfiltered
else:
  unfiltered

(similarity mode can apply score-threshold filtering;
MMR mode can apply optional reranking)
```

## Why this helps

- Keeps prompts inside context limits.
- Reduces noisy or repetitive RAG injection.
- Preserves conversation continuity while avoiding overgrowth.
