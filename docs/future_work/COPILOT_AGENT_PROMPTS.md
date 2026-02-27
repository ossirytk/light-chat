# Copilot Agent Prompts — RAG File Normalization

Use these prompts directly in Copilot Agent mode.

## How to use these prompts (quick workflow)

1. Open Copilot Chat in **Agent mode**.
2. Copy one prompt section from this file.
3. Paste it into Agent mode and run.
4. Let the agent complete edits and validation commands.
5. Review changed files in Source Control.
6. If needed, run a second pass with the same prompt plus a short follow-up constraint.

Recommended follow-up checks after agent completion:

- `uv run ruff format .`
- `uv run ruff check .`
- Quick manual spot-check of changed files in `rag_data/`.

Tips for best results:

- Run one prompt at a time (context first, then message examples).
- Keep `shodan` files as reference baselines; avoid changing them.
- If output drifts, rerun with: "Preserve current facts, only normalize structure/style."

---

## Prompt 1 — Normalize context files to match shodan.txt style

Use this repository’s existing style and tooling to normalize character context files in `rag_data` so they match the structure and readability of `rag_data/shodan.txt`.

Goal:
- Convert target context documents into a consistent markdown format with:
  - metadata header comment on line 1
  - clear H2 sections
  - optional H3 subsections for timeline/relationships
  - short fact-focused lines (avoid long encyclopedia paragraphs)
- Preserve factual meaning; do not invent lore.
- Remove duplicate paragraphs, boilerplate residue, and noisy caption-like lines.
- Keep one-character-per-file focus.

Required format baseline:
- Follow the sectioning style used in `rag_data/shodan.txt`.
- Keep concise sentence structure similar to shodan context style.

Files:
- Primary target: `rag_data/leonardo_da_vinci.txt`
- Then apply same pattern to any other raw context files in `rag_data` that are not yet normalized.

Constraints:
- Keep edits minimal and deterministic.
- Do not change shodan files unless needed for consistency checks.
- Use project tooling: `uv` + `ruff`.
- Run:
  - `uv run ruff format .`
  - `uv run ruff check .`
- If unrelated lint issues exist, report them but do not over-fix unrelated files.

Do-not-modify paths:
- `rag_data/shodan.txt`
- `rag_data/shodan_message_examples.txt`

Stop-and-ask conditions:
- If source text contains conflicting facts that require choosing one version.
- If a transformation would require adding information not present in the source.

Definition of Done:
- Metadata header exists on line 1.
- Document is split into clear `##` sections (and optional `###` for timeline/relationships).
- Long encyclopedia paragraphs are replaced with concise fact-focused lines.
- Obvious duplicates/noise are removed.
- Ruff format/check completed and reported.

Deliverables:
- Updated normalized context files.
- Short summary of what changed per file.
- Any remaining manual-review items (if ambiguity exists).

Output template:
- `Files changed:`
- `Structure checks:` (headers/sections present)
- `Validation:` (`ruff format`, `ruff check`)
- `Manual review notes:`

## Prompt 2 — Normalize message example files to match shodan_message_examples.txt style

Normalize message example files in `rag_data` so they follow the same design pattern as `rag_data/shodan_message_examples.txt`.

Goal:
- Enforce a consistent example format:
  - metadata header comment on top
  - scenario label line starting with `#` (for example greeting, threatening, technical explanation)
  - explicit `User:` line
  - explicit `CharacterName:` response line
  - short, reusable 2–4 line exchanges
- Ensure examples are character-accurate in tone and worldview.
- Remove quote dumps or unlabeled one-liners that are not in dialogue format.

Files:
- Primary target: `rag_data/leonardo_da_vinci_message_examples.txt`
- Then apply same format checks to other `*_message_examples.txt` files in `rag_data`.

Content quality requirements:
- Include scenario diversity (greeting, philosophy, explanation, reflection, directive where relevant).
- Avoid very long monologues.
- Keep language clean and consistent with corresponding context file.

Constraints:
- Preserve existing high-quality shodan examples as style reference.
- Do not invent out-of-character claims.
- Keep deterministic, reviewable edits.

Do-not-modify paths:
- `rag_data/shodan_message_examples.txt`

Stop-and-ask conditions:
- If character voice is unclear due to missing context.
- If an edit would require creating lore that is not present in context files.

Validation:
- Confirm each example block has:
  - one scenario label
  - one User line
  - one character response line
- Run:
  - `uv run ruff format .`
  - `uv run ruff check .`

Definition of Done:
- Metadata header exists at top of each normalized example file.
- Every block has one scenario label, one `User:` line, and one character response line.
- Blocks are concise and reusable (target 2–4 lines per exchange block).
- Scenario coverage is reasonably diverse.
- Ruff format/check completed and reported.

Deliverables:
- Updated message-example files in consistent format.
- Brief report of counts per file (scenario labels and User lines).
- Note any files needing manual curation.

Output template:
- `Files changed:`
- `Block format counts:` (scenario labels, `User:` lines)
- `Validation:` (`ruff format`, `ruff check`)
- `Manual review notes:`
