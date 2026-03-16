# conversation_template.json

Last verified: 2026-03-12

File: `configs/conversation_template.json`

## Purpose

Defines the prompt template used by non-mistral model paths.

For `MODEL_TYPE == mistral`, runtime uses a dedicated prompt builder in `ConversationPromptHistoryMixin` (via `ConversationManager`) and does not rely on this template for the final prompt string.

## Template Type

- `_type`: `prompt`
- `input_variables`: variable placeholders consumed by LangChain prompt rendering

## Input Variables

Current template expects:

- `llama_instruction`
- `character`
- `llama_input`
- `description`
- `scenario`
- `mes_example`
- `vector_context`
- `history`
- `input`
- `llama_response`
- `llama_endtoken`

## Prompt Structure (high-level)

1. System/persona instruction
2. Character description and scenario
3. Message examples
4. Retrieved vector context block
5. Conversation history
6. Current user turn
7. Assistant response prefix

## Safe Editing Guidelines

- Keep placeholder names aligned with runtime variables.
- Preserve role/instruction boundaries for model compatibility.
- Avoid inserting user-turn markers in assistant instructions that could conflict with response guards.
- If you add/remove variables, update runtime prompt input assembly accordingly.

## Related Runtime Code

- Prompt loading: `ConversationModelSetupMixin._load_prompt_template`
- Prompt inputs for template path: `ConversationPromptHistoryMixin._build_conversation_chain`
