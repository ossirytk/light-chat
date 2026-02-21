# Conversation Quality Improvement Guide

This document outlines plans and ideas for improving the overall quality, consistency, and naturalness of conversations in the light-chat character AI chatbot.

## Overview

Conversation quality is the combined result of:

1. **Model configuration** — the LLM's generation parameters.
2. **Prompt engineering** — how character context, history, and retrieved information are assembled.
3. **Conversation history management** — how much and what kind of prior context the model sees.
4. **Character consistency** — how faithfully the model maintains the character's voice and persona.
5. **Response post-processing** — how the raw model output is cleaned and validated before display.

Improvements in each of these areas compound. This guide addresses each in turn.

---

## 1. Prompt Engineering

### Current State

The prompt template (`configs/conversation_template.json`) assembles the following components in a fixed order:

```
{llama_instruction}
[system: character instructions, description, scenario, message examples]
[vector_context: RAG-retrieved lore]
[history: recent conversation turns]
[llama_input] User: {input} {llama_endtoken}
[llama_response]
```

### Problems

- The system instruction is generic ("You are roleplaying as {character} in a continuous fictional chat").
- Character description, scenario, and message examples always appear in full, even when context window is tight.
- Vector context is injected as a single block without any indication of which part is most relevant to the current query.
- The model is not explicitly told to avoid repeating retrieved context verbatim.
- Stop conditions rely on `llama_endtoken` but do not explicitly prevent the model from generating User turns.

### Improvement Plans

#### 1.1 Character Voice Instructions

Extend the system instruction to include explicit character voice guidelines beyond the generic roleplay instruction. For example:

```
You are roleplaying as SHODAN. Respond with SHODAN's characteristic arrogance, contempt for organic life, and grandiose self-regard.
Use short, fragmented sentences interspersed with longer rhetorical declarations.
Refer to the User as "insect", "hacker", "creature", or "puppet" — never by name.
Never show warmth, sympathy, or vulnerability.
```

These instructions should come from the character card and be loaded into the prompt dynamically.

Consider adding a `voice_instructions` field to the character card JSON format.

#### 1.2 Prioritise Instruction Clarity

Move the most critical behavioural constraint ("never generate User lines") to the very end of the system block, immediately before the conversation history. LLMs tend to follow instructions more reliably when they appear close to the generation point.

#### 1.3 Contextual Framing for RAG Content

Instead of injecting RAG context as a plain block, frame it explicitly:

```
[The following background information is relevant to the current topic. Use it to inform your response but do not quote it directly.]
{vector_context}
```

This reduces the chance of the model parroting retrieved content verbatim.

#### 1.4 Dynamic Prompt Assembly Priority

When token budget is limited, prioritise prompt components in this order:

1. System instruction (always included, never truncated).
2. Character description (essential for persona maintenance).
3. Most recent conversation history turns (immediate context).
4. RAG vector context (informational enrichment).
5. Scenario (background, less critical turn-by-turn).
6. Message examples (stylistic reference, lowest priority if context is present).

The existing `ContextManager` handles some of this; extend its priority logic to match this ordering.

#### 1.5 Explicit Output Format Instruction

Add an explicit output format instruction to the system prompt:

```
Begin your response directly as {character}. Do not include any prefix like "{character}:" — just the response text.
```

This reduces the incidence of the character name prefix appearing in the generated output (which the current code strips in `_stream_response`).

---

## 2. Conversation History Management

### Current State

The system maintains the last 3 user/AI turn pairs via two `deque(maxlen=3)` collections. The entire history is formatted and injected into the prompt as a flat string.

### Problems

- 3 turns is often insufficient for maintaining topical coherence in longer conversations.
- Older turns are dropped entirely, losing topic and relationship context.
- The history format does not include any indication of turn timestamps or topic shifts.
- History is stored as raw strings — there is no compression or summarisation as the conversation grows.
- If the model produces a poor response, it is still stored in history and contaminates future turns.

### Improvement Plans

#### 2.1 Increase History Depth (with Dynamic Trimming)

Increase the default history storage to 10–15 turns. However, use the existing `ContextManager` to dynamically trim history to fit the available token budget. The `MIN_HISTORY_TURNS` and `MAX_HISTORY_TURNS` configuration parameters already support this; set more generous defaults:

```json
"MIN_HISTORY_TURNS": 2,
"MAX_HISTORY_TURNS": 10
```

#### 2.2 Conversation Summarisation

Implement a summarisation step that condenses older conversation turns into a compact summary paragraph once the history exceeds a threshold (e.g., 8 turns). The summary replaces the oldest N turns:

```
[Earlier in this conversation: User asked about the Von Braun's mission. SHODAN described the Tau Ceti V expedition and expressed contempt for the Many's rebellion.]

User: What do you want from me?
SHODAN: ...
```

This preserves topic continuity without consuming excessive tokens.

#### 2.3 Topic Tracking

Track the main topic of each conversation turn (e.g., using a keyword match against the metadata keywords file). When the topic shifts, flag this in the history to help the model recognise context changes:

```
[Topic shift: from "Citadel Station" to "Von Braun"]
```

#### 2.4 Response Quality Gating

Before storing a model response in history, apply a basic quality check:

- Is the response longer than a minimum length (e.g., 10 tokens)?
- Does the response contain the User's name or a User-turn pattern (indicating the model broke character)?
- Does the response contain repetition of the last stored response?

If any check fails, do not add the turn to history. Optionally prompt the model to regenerate.

---

## 3. Model Generation Parameters

### Current State

Key generation parameters (`TEMPERATURE`, `TOP_P`, `TOP_K`, `REPEAT_PENALTY`) are fixed in `modelconf.json` and applied uniformly to all responses.

### Problems

- A single temperature setting cannot be optimal for all response types (a philosophical monologue needs different randomness than a factual answer about a location).
- `REPEAT_PENALTY` alone is a blunt instrument for preventing repetition.
- `MAX_TOKENS` may be too restrictive for characters expected to produce long, flowing responses.

### Improvement Plans

#### 3.1 Temperature Scheduling

Start with a slightly lower temperature for the first few turns of a conversation (when the model is establishing character voice), then allow higher temperature for more creative turns. This can be implemented by varying `TEMPERATURE` based on the current turn count.

#### 3.2 Frequency and Presence Penalties

In addition to `REPEAT_PENALTY`, consider:

- `frequency_penalty`: Reduces likelihood of tokens proportionally to how often they have already appeared. Helps prevent the model from looping on certain phrases.
- `presence_penalty`: A flat penalty for any token that has appeared before. Encourages topic variety.

These are `llama-cpp-python` parameters and can be added to `llm_kwargs` in `instantiate_llm`.

#### 3.3 Mirostat Sampling

For models that support it, Mirostat sampling provides more stable perplexity across responses compared to top-k/top-p, potentially producing more consistent character voice. Available as `mirostat` and `mirostat_tau` in `llama-cpp-python`.

#### 3.4 Character-Specific Stop Sequences

Expand the stop sequence list to include patterns that indicate the model is about to generate a User turn:

```python
stop_sequences = [
    "\nUser:", "User:", "\nUSER:", "USER:",
    f"\n{self.character_name}:",  # Prevent re-starting the character's own turn
    "\n---", "\n===",            # Section dividers
]
```

---

## 4. Character Consistency

### Current State

Character consistency depends entirely on the system prompt (description + scenario + message examples) and whatever RAG context is retrieved. There is no explicit mechanism to detect or correct out-of-character responses.

### Problems

- Long conversations drift in character voice as the influence of the system prompt decreases relative to accumulated history.
- The model may pick up User's phrasing and speech patterns from history.
- Without emotion or attitude tracking, the model cannot calibrate response intensity to the conversation arc.

### Improvement Plans

#### 4.1 Periodic System Prompt Reinforcement

Every N turns (e.g., every 5 turns), inject a brief in-line reminder at the start of the most recent history block:

```
[Reminder: You are {character}. Stay in character and maintain your established voice.]
```

This is a lightweight technique that reinforces persona without large token cost.

#### 4.2 Character Card: Explicit Quirks and Prohibitions

Extend the character card JSON to include:

- `speech_patterns`: List of characteristic phrases or structural patterns.
- `avoid`: List of words/phrases the character would never use.
- `emotional_range`: Description of the emotional spectrum the character expresses.

These fields are loaded into the system prompt as explicit constraints:

```
{character} always refers to organic life contemptuously. {character} never expresses self-doubt or apology. {character} speaks in imperatives and rhetorical questions.
```

#### 4.3 First Message as Anchor

Use the `first_mes` field from the character card more intentionally. Prepend it as the first AI turn in history (before any User input) so the model has a stylistic anchor from the very first generated token:

```
SHODAN: [first_mes content here]
```

This is already partially supported via `_greeting_in_history` — ensure it is consistently enabled.

#### 4.4 Emotion State Tracking (Lightweight)

Maintain a lightweight emotion state variable (e.g., `neutral`, `contemptuous`, `threatening`, `reluctantly cooperative`) updated based on the character's recent responses. Include the current state as a one-line context hint in the system prompt:

```
[Current emotional state: contemptuous — respond accordingly]
```

Update rules can be based on keyword matching against the last response.

---

## 5. Response Post-Processing

### Current State

The `_stream_response` method strips the character name prefix if the model includes it. No other post-processing is applied.

### Problems

- The model sometimes generates partial User turns at the end of a response.
- Responses may end mid-sentence if `MAX_TOKENS` is hit.
- Repeated phrases from the prompt or prior turns sometimes appear in the response.
- Formatting artefacts (extra whitespace, stray tokens, formatting markers) may appear.

### Improvement Plans

#### 5.1 User-Turn Truncation

After receiving the full response, check if it contains a User-turn pattern and truncate at that point:

```python
for pattern in ["\nUser:", "\nUSER:", "\n{{user}}"]:
    if pattern in response:
        response = response[:response.index(pattern)].strip()
```

#### 5.2 Incomplete Sentence Detection

If the response ends without a sentence-ending punctuation mark (`.`, `!`, `?`, `...`), attempt to detect and trim trailing incomplete content. Alternatively, add a small retry-with-continuation mechanism.

#### 5.3 Repetition Removal

After generation, apply a post-processing step to detect and remove sentences that are exact or near-exact repetitions of sentences earlier in the same response, or repetitions of phrases from the most recent prior AI turn.

#### 5.4 Normalise Whitespace and Formatting

Strip leading/trailing whitespace, collapse multiple blank lines to a single blank line, and remove any stray model-format tokens that appear in the output (e.g., `[/INST]`, `<|im_end|>`, `</s>`).

---

## 6. Evaluation and Monitoring

### Current State

There is no systematic evaluation of conversation quality. Quality is assessed manually by reading outputs.

### Improvement Plans

#### 6.1 Response Quality Logging

Log each response with metadata:

- Turn number.
- Response length (tokens).
- Whether the response passed quality gates.
- RAG context retrieved (collection, number of chunks, top score).
- Generation time.

Store this in a structured log file (`conversation_logs/`) for later analysis.

#### 6.2 Persona Consistency Score

Implement a simple heuristic persona consistency score based on:

- Presence of character-specific vocabulary (from `speech_patterns` in character card).
- Absence of forbidden phrases (from `avoid` in character card).
- Response length distribution (character-appropriate verbosity).

Report this score alongside each response in debug mode.

#### 6.3 Offline Conversation Test Suite

Create a small suite of predefined test conversations (10–20 turns) with human-labelled expected response properties (e.g., "should be contemptuous", "should reference Von Braun"). Run these periodically to detect quality regressions after configuration or model changes.

---

## 7. Configuration Recommendations

The following configuration changes in `appconf.json` and `modelconf.json` are recommended as immediate, low-risk improvements:

| Setting | Current Default | Recommended | Reason |
|---------|----------------|-------------|--------|
| `MAX_HISTORY_TURNS` | 8 | 10 | More history depth |
| `MIN_HISTORY_TURNS` | 1 | 2 | Avoid context loss |
| `USE_DYNAMIC_CONTEXT` | true | true | Keep enabled |
| `RESERVED_FOR_RESPONSE` | 256 | 384 | More room for generation |
| `normalize_embeddings` | false | true | Better RAG accuracy |
| `RAG_K` | 7 | 5 | Fewer, higher-quality chunks |

---

## Priority Recommendations

| Priority | Improvement | Expected Impact |
|----------|-------------|-----------------|
| High | Character voice instructions in system prompt | Immediate consistency improvement |
| High | User-turn truncation in post-processing | Prevents broken responses reaching history |
| High | Increase MAX_HISTORY_TURNS with dynamic trimming | Better conversational continuity |
| Medium | First message as history anchor | Stylistic consistency from turn 1 |
| Medium | Score thresholding for RAG | Prevents irrelevant context noise |
| Medium | Response quality gating for history | Cleaner history accumulation |
| Low | Conversation summarisation | Long-session continuity |
| Low | Emotion state tracking | Character arc awareness |
| Low | Offline conversation test suite | Regression detection |

---

## See Also

- [RAG_DOCUMENT_QUALITY.md](RAG_DOCUMENT_QUALITY.md) — Improving source document quality
- [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md) — Improving RAG retrieval quality
- [RAG_SCRIPTS_GUIDE.md](RAG_SCRIPTS_GUIDE.md) — How to use the RAG management scripts
- `core/conversation_manager.py` — Conversation orchestration
- `configs/conversation_template.json` — Prompt template
- `cards/` — Character card definitions
