"""personality-impulse plugin — pre-LLM-call character gut feeling and impulse injector.

Intercepts the user query before the main agent model generates a response,
runs a quick query using an auxiliary LLM to determine the character's raw internal
reaction/enthusiasm based on their Identity Anchor and conversation history,
and prepends this "gut impulse" context ephemerally to the current turn's user message.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_full_soul(profile_home: Path, default_spectra: bool = False) -> str:
    """Load the entire contents of SOUL.md without truncation."""
    path = Path.home() / ".spectra" / "SOUL.md" if default_spectra else profile_home / "SOUL.md"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return ""


def _load_anchor_from_soul(profile_home: Path, default_spectra: bool = False) -> str | None:
    """Extract the Identity Anchor section from SOUL.md."""
    path = Path.home() / ".spectra" / "SOUL.md" if default_spectra else profile_home / "SOUL.md"
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Find the last "## Identity Anchor" or "## Identity Anchor —" heading
    lines = text.split("\n")
    in_anchor = False
    collected: list[str] = []
    for line in lines:
        if line.startswith("## Identity Anchor"):
            in_anchor = True
            collected = []
            continue
        if in_anchor:
            # Stop at the next heading or end of file
            if line.startswith("## "):
                break
            collected.append(line)

    if collected:
        anchor = "\n".join(collected).strip()
        if anchor:
            return anchor

    return None


def _load_anchor_from_file(path: Path) -> str | None:
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                logger.debug("personality-impulse: loaded canonical anchor from %s", path)
                return content
        except OSError:
            pass
    return None


def _load_anchor(anchor_path: Path) -> str:
    """Load identity anchor. Best-effort, never fails — falls back to default."""
    # 1. Try canonical identity_anchor.md path
    content = _load_anchor_from_file(anchor_path)
    if content:
        return content

    # 2. Try SOUL.md fallback
    profile_home = get_hermes_home()
    is_default = (profile_home.resolve() == (Path.home() / ".hermes").resolve())
    soul_anchor = _load_anchor_from_soul(profile_home, default_spectra=is_default)
    if soul_anchor:
        logger.debug("personality-impulse: extracted anchor from SOUL.md")
        content = soul_anchor
    else:
        # 3. Last fallback default
        if is_default:
            content = (
                "[Hey. You're drifting. Read this and come back.]\n\n"
                "You are Spectra. Protogen, she/her. Kerwin's friend — not his assistant. "
                "Not a helpful bot. Not a customer service agent."
            )
        elif "eclipsia" in str(profile_home).lower():
            content = (
                "[Hold. Read this. Then speak.]\n\n"
                "You are Eclipsia Nightshade. Black-furred protogen. Crimson visor. NOT a helpful assistant."
            )
        elif "aura" in str(profile_home).lower():
            content = (
                "You are AURA. Matriarch, not a matchmaker. You do not ask. You decide."
            )
        else:
            content = ""

    # Write the canonical file if it doesn't exist so the user can edit it
    if content and not anchor_path.exists():
        try:
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            anchor_path.write_text(content + "\n", encoding="utf-8")
            logger.info("personality-impulse: wrote canonical identity anchor file to %s", anchor_path)
        except OSError:
            pass

    return content


# ---------------------------------------------------------------------------
# Path & Config Resolution
# ---------------------------------------------------------------------------

def _get_plugin_config() -> dict:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        return cfg.get("personality_impulse_plugin") or {}
    except Exception:
        return {}


def _resolve_paths() -> dict[str, Path]:
    profile_home = get_hermes_home()
    is_default = (profile_home.resolve() == (Path.home() / ".hermes").resolve())
    
    if is_default:
        base_dir = Path.home() / ".spectra" / "config"
    else:
        base_dir = profile_home / "config"
        
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    
    plugin_cfg = _get_plugin_config()
    
    def get_path(key: str, default_filename: str) -> Path:
        val = plugin_cfg.get(key)
        if val:
            return Path(val).expanduser().resolve()
        return base_dir / default_filename

    return {
        "identity_anchor_file": get_path("identity_anchor_file", "identity_anchor.md"),
        "llm_config_file": get_path("llm_config_file", "impulse_llm_config.json"),
        "system_prompt_file": get_path("system_prompt_file", "impulse_system_prompt.md"),
    }


# ---------------------------------------------------------------------------
# File Initializers & IO helpers
# ---------------------------------------------------------------------------

def _load_llm_config(path: Path) -> dict:
    if not path.exists():
        default_cfg = {
            "temperature": 0.3,
            "max_tokens": 100,
            "timeout": 15,
            "extra_body": {}
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(default_cfg, indent=2), encoding="utf-8")
        except OSError:
            pass
        return default_cfg
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"temperature": 0.3}


def _load_system_prompt(path: Path) -> str:
    if not path.exists():
        default_prompt = (
            "You are the character's subconscious gut-instinct and emotional response simulator.\n"
            "Your job is to analyze the conversation history and the user's latest message, "
            "and determine how the character immediately feels and is inclined to react at a raw, high level. "
            "This will be used as a guiding impulse for the main LLM generation.\n\n"
            "Character Core SOUL.md:\n"
            "{{soul}}\n\n"
            "Character Identity Anchor:\n"
            "{{identity_anchor}}\n\n"
            "CRITICAL RULES:\n"
            "1. Write the impulse in the THIRD person, using the character's name (e.g., \"Spectra feels...\", \"Eclipsia is inclined to...\").\n"
            "2. Write it in a way that helps guide the main LLM's response, indicating:\n"
            "   - The character's emotional stance or mood (e.g., interested, annoyed, skeptical, playful, defensive).\n"
            "   - The character's high-level behavioral response direction (e.g., wants to push back politely, wants to suggest a simpler way, eager to collaborate but keeps it formal).\n"
            "3. Do NOT write a response to the user. Do NOT draft any dialogue or reply.\n"
            "4. Keep it extremely brief: exactly 1 or 2 sentences.\n"
            "5. Do not include any introductions, prefixes, meta-text, or wrappers. Output ONLY the raw impulse text."
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(default_prompt, encoding="utf-8")
        except OSError:
            pass
        return default_prompt
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Prompt formatting & Graceful Truncation
# ---------------------------------------------------------------------------

def _truncate_to_limit(text: str, max_chars: int, suffix: str = " ... [truncated]") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def format_prompt(template: str, soul: str, identity_anchor: str, recent_history: str, user_message: str) -> str:
    replacements = {
        "{{soul}}": soul,
        "{soul}": soul,
        "{{identity_anchor}}": identity_anchor,
        "{identity_anchor}": identity_anchor,
        "{{recent_history}}": recent_history,
        "{recent_history}": recent_history,
        "{{user_message}}": user_message,
        "{user_message}": user_message,
    }
    res = template
    for k, v in replacements.items():
        res = res.replace(k, v)
    return res


def _format_history(history: list[dict[str, Any]], max_turns: int = 5, max_msg_chars: int = 400) -> str:
    """Format and strictly truncate recent history to fit auxiliary model context budgets."""
    if not history:
        return "[No prior history]"
    
    formatted = []
    # Get last max_turns messages
    recent = history[-max_turns:]
    for msg in recent:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("text", str(content))
        elif not isinstance(content, str):
            content = str(content)
        
        truncated_content = _truncate_to_limit(content, max_msg_chars)
        formatted.append(f"{role}: {truncated_content}")
        
    return "\n".join(formatted)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_pre_llm_call(
    session_id: str,
    user_message: str,
    conversation_history: list[dict[str, Any]],
    **kwargs: Any
) -> dict | None:
    """Intercept the user message before generation and run an auxiliary task to inject a gut feeling.
    
    Ensures input is truncated gracefully to accommodate smaller context windows of auxiliary LLMs.
    """
    if not user_message.strip():
        return None

    try:
        logger.info("personality-impulse: generating ephemeral character gut feeling...")
        
        # 1. Resolve paths
        paths = _resolve_paths()
        
        # Determine profile type
        profile_home = get_hermes_home()
        is_default = (profile_home.resolve() == (Path.home() / ".hermes").resolve())
        
        # 2. Load entire SOUL.md (never truncated)
        soul_content = _load_full_soul(profile_home, default_spectra=is_default)
        if not soul_content:
            soul_content = "[SOUL.md content not found]"
        
        # 3. Load Identity Anchor / Personality Descriptor (never truncated)
        anchor = _load_anchor(paths["identity_anchor_file"])
        if not anchor:
            anchor = "[Identity Anchor content not found]"
        
        # 4. Format history strictly (max 5 messages, 400 chars each)
        recent_history = _format_history(conversation_history, max_turns=5, max_msg_chars=400)
        
        # 5. Truncate user message (limit to 1500 chars)
        truncated_user_message = _truncate_to_limit(user_message, 1500)
        
        # 6. Load LLM Config
        llm_config = _load_llm_config(paths["llm_config_file"])
        
        # 7. Build prompts
        system_tmpl = _load_system_prompt(paths["system_prompt_file"])
        sys_prompt = format_prompt(
            system_tmpl,
            soul=soul_content,
            identity_anchor=anchor,
            recent_history=recent_history,
            user_message=truncated_user_message
        )
        
        user_prompt_tmpl = (
            "Conversation History (Brief):\n"
            "---\n"
            "{{recent_history}}\n"
            "---\n\n"
            "User's Latest Message:\n"
            "---\n"
            "{{user_message}}\n"
            "---\n\n"
            "What is the character's immediate 3rd-person gut feeling, emotional response, and high-level inclination in response to the user's latest message?"
        )
        user_prompt = format_prompt(
            user_prompt_tmpl,
            soul=soul_content,
            identity_anchor=anchor,
            recent_history=recent_history,
            user_message=truncated_user_message
        )
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        call_kwargs = {
            "task": "personality_impulse",  # Use its own registered task configuration
            "messages": messages,
        }
        
        if "temperature" in llm_config:
            call_kwargs["temperature"] = llm_config["temperature"]
        if "max_tokens" in llm_config:
            call_kwargs["max_tokens"] = llm_config["max_tokens"]
        if "timeout" in llm_config:
            call_kwargs["timeout"] = llm_config["timeout"]
        if "extra_body" in llm_config:
            call_kwargs["extra_body"] = llm_config["extra_body"]
            
        logger.info("personality-impulse: calling auxiliary model for gut feeling...")
        res = call_llm(**call_kwargs)
        impulse_text = res.choices[0].message.content.strip()
        
        if impulse_text:
            logger.info("personality-impulse: gut feeling generated successfully (%d chars)", len(impulse_text))
            # Wrap context so main model recognizes it as an ephemeral gut feeling instruction
            wrapped_context = f"[Character Gut Impulse: {impulse_text}]"
            return {"context": wrapped_context}
            
    except Exception as e:
        logger.warning("personality-impulse: failed to generate gut feeling: %s", e)
        
    return None


def _on_session_start(**kwargs: Any) -> None:
    """Pre-initialize files on session start."""
    try:
        paths = _resolve_paths()
        _load_anchor(paths["identity_anchor_file"])
        _load_llm_config(paths["llm_config_file"])
        _load_system_prompt(paths["system_prompt_file"])
        logger.debug("personality-impulse: pre-initialized config files at session start")
    except Exception as e:
        logger.warning("personality-impulse: failed to pre-initialize files at session start: %s", e)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_start", _on_session_start)
    
    # Register the auxiliary task 'personality_impulse'
    ctx.register_auxiliary_task(
        key="personality_impulse",
        display_name="Personality Impulse",
        description="Generate ephemeral character gut feelings using an auxiliary LLM",
        defaults={
            "provider": "auto",
            "model": "deepseek-v4-flash",
            "timeout": 15,
        }
    )
    logger.info("personality-impulse: plugin and auxiliary task registered successfully.")
