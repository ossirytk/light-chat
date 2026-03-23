# Message Examples Sectioning Style Guide

Last updated: 2026-03-16

This document defines the authoritative style rules for character message example files (`*_message_examples.txt`). These files provide context windows showing how a character speaks, helping RAG retrieval and LLM generation maintain character voice.

## File Structure

### 1. Header (Required)

Every message examples file must begin with an HTML metadata comment:

```html
<!-- character: CHARACTER_NAME | source: SOURCE_REFERENCE | version: VERSION_NUM | edited: DATE -->
```

**Examples:**
```html
<!-- character: Leonardo da Vinci | source: Historical writings and notes | version: 2 | edited: 2024-03-15 -->
<!-- character: Shodan | source: System Shock dialogue logs | version: 1 | edited: 2024-11-22 -->
```

**Rules:**
- Must be the first line (after optional blank lines)
- Fields: `character`, `source`, `version`, `edited` (all required)
- Date format: YYYY-MM-DD
- This line is **stripped during chunking** and not embedded as retrievable content

### 2. Optional Preamble

After the header, a brief intro section may appear (max 3-4 lines):

```
[OPTIONAL INTRO TEXT]

Next section begins after blank line...
```

**Rules:**
- Max 200 characters
- Kept in document, but prefaced with metadata context to avoid leakage
- Blank line after preamble before first message pair

### 3. Message Pairs (Required)

Message pairs follow a consistent format:

```
[USER]: User message here...

[ASSISTANT]: Assistant response here...

```

**Rules:**
- Each message on separate lines (no inline)
- Labels: `[USER]:` and `[ASSISTANT]:` (uppercase, square brackets, colon)
- One blank line between pairs
- No mixed styles (cannot have both `User:` and `[USER]:` in same file)
- Messages can be multi-line; entire block until next label is treated as the message

**Valid Example:**
```
[USER]: What is your most famous invention?

[ASSISTANT]: The printing press stands foremost among my contributions to mankind.
It enabled the democratization of knowledge...

[USER]: Tell me about your approach to Renaissance art.

[ASSISTANT]: Rather than viewing disciplines in isolation, I studied light, geometry,
and the human form holistically. By understanding nature's laws...
```

### 4. Section Breaks (Optional)

To separate thematic message groups, use a section delimiter:

```
---

```

**Rules:**
- Three hyphens, standalone line
- One blank line before and after
- Optional; helps readability but not required for parsing

### 5. Example: Complete Valid File

```html
<!-- character: Leonardo da Vinci | source: Notebook excerpts | version: 1 | edited: 2024-03-15 -->

Leonardo da Vinci reflects on art, science, and human potential with Renaissance curiosity.

[USER]: How do you balance art and science?

[ASSISTANT]: Art and science are not separate pursuits—they are one. To paint the human form
requires understanding anatomy. To design machines demands knowledge of physics. The greatest
works emerge when both mind and hand are trained equally.

[USER]: What drives your curiosity?

[ASSISTANT]: Curiosity is the root of all knowledge, and knowledge is power. Every shadow cast
by candlelight teaches something of optics. Every river's flow reveals principles that govern
machines. The world speaks constantly; one need only listen and observe.

---

[USER]: Describe your workshop.

[ASSISTANT]: My workshop is both chaos and order. Sketches cover every surface—designs for
machines, anatomical studies, architectural plans. Paint, clay, and tools intermingle. But within
this apparent disorder lives method. Each sketch builds upon previous observations...
```

## Consistency Rules for Linting

The automated `lint-message-examples` tool validates:

1. **Header presence**: File must have HTML comment header ✓  
2. **Header format**: All four fields (`character`, `source`, `version`, `edited`) must be present ✓  
3. **Message pair format**: All messages follow `[USER]:` / `[ASSISTANT]:` pattern with no mixing ✓  
4. **Delimiter consistency**: No mixing of delimiter styles (use `---` or nothing, not mixed) ✓  
5. **Blank line separation**: Message pairs separated by blank lines ✓  

## Common Violations & Fixes

| Violation | Example | Fix |
|-----------|---------|-----|
| Missing header | File starts with `[USER]:` | Add metadata comment as first line |
| Header incomplete | `<!-- character: Name -->` | Add missing fields: `source:`, `version:`, `edited:` |
| Mixed label styles | Mix of `[USER]:` and `User:` | Normalize to `[USER]:` consistently |
| No blank lines | Two pairs back-to-back | Insert blank line between pairs |
| Wrong delimiter format | Dashes at different lengths (`--` or `----`) | Use exactly three hyphens: `---` |

## Rationale

- **HTML comment header**: Prevents metadata from being indexed as content; standard format easy to parse
- **`[LABEL]:` format**: Unambiguous parsing; case-insensitive searching works reliably
- **Blank line delimiters**: Makes diffs readable; helps chunking algorithms recognize boundaries
- **Consistency**: Enables deterministic linting and auto-fixing; reduces merge conflicts

## Future Enhancements

- Optional `[CONTEXT]:` label for scene-setting information between message pairs
- Tagging system for message topics (e.g., `[USER metadata="philosophy"]`)
- Automatic generation of fixture test cases from message pairs
