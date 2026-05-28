# personality-impulse

A Hermes plugin that injects a short, ephemeral character gut feeling and response impulse before every LLM call.

## Description

The `personality-impulse` plugin runs an auxiliary LLM call using the `pre_llm_call` hook.
It analyzes:
1. The character's Identity Anchor (loaded from `identity_anchor.md` or `SOUL.md`).
2. The recent conversation history (up to the last 6 turns).
3. The user's latest incoming message.

From this context, the auxiliary LLM generates a 1-to-2 sentence first-person internal impulse (e.g. `[Character Gut Impulse: I feel excited about building this but want to push back on using that complex framework. I'm inclined to respond with a dry joke.]`).
This impulse is injected ephemerally as context prepended to the user's message during the API call, so it influences the main generation but does **not** persist in the session's chat history.

## Configuration Files

The plugin automatically initializes the following files in the character config folder (`~/.spectra/config` or profile config) on session start:
- `identity_anchor.md`: The character's core personality constraints.
- `impulse_llm_config.json`: Options for the auxiliary task API call.
- `impulse_system_prompt.md`: The system instructions used to format the gut feeling generator.
