"""FastAPI + Jinja2 + HTMX web chat interface for light-chat."""

import asyncio
import contextlib
import html
import io
import json
import threading
import time
import uuid
import zipfile
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel

from core import rag_manager
from core.config import configure_logging, load_app_config, load_rag_script_config
from core.conversation_manager import ConversationManager
from core.job_queue import JobStore
from core.preset_profiles import ProfileStore


class StreamRequest(BaseModel):
    """Streaming request payload."""

    message: str
    continue_mode: bool = False


_INVALID_STEM_HTML = (
    "<p class='text-error'>"
    "Invalid stem: only letters, digits, underscores, and hyphens are allowed."
    "</p>"
)
_INVALID_COLL_HTML = (
    "<p class='text-error'>"
    "Invalid collection name: only letters, digits, underscores, and hyphens are allowed."
    "</p>"
)
_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB


class ChatRuntime:
    """Holds shared chat runtime state for web requests."""

    MAX_RETRIEVAL_HISTORY_ENTRIES: int = 40

    def __init__(self, manager: ConversationManager) -> None:
        self.manager = manager
        self.lock = asyncio.Lock()
        self.started_at = time.time()
        self.session_dir = Path("logs") / "web_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.ui_messages: list[dict[str, str]] = []
        self.retrieval_history: list[dict[str, object]] = []
        self.reset_ui_messages()

    def reset_ui_messages(self) -> None:
        self.ui_messages = []
        if self.manager.first_message:
            self.ui_messages.append({"role": "assistant", "content": self.manager.first_message})


templates = Jinja2Templates(directory="templates")

CONTINUE_PROMPT = (
    "Continue from exactly where your previous answer stopped. "
    "Do not repeat prior text. Start with the next unfinished sentence."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialize and cleanup shared app resources."""
    app_config = load_app_config()
    configure_logging(app_config)
    manager = await asyncio.to_thread(ConversationManager)
    app.state.chat_runtime = ChatRuntime(manager)
    app.state.profile_store = ProfileStore(Path("configs") / "profiles.json")
    app.state.rag_config = load_rag_script_config(app_config)
    app.state.job_store = JobStore()
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            del app.state.chat_runtime


app = FastAPI(title="light-chat web", lifespan=lifespan)


def _get_runtime(request: Request) -> ChatRuntime:
    return request.app.state.chat_runtime


def _get_profile_store(request: Request) -> ProfileStore:
    return request.app.state.profile_store


def _get_rag_config(request: Request) -> object:
    return request.app.state.rag_config


def _get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _render_chat_log(messages: list[dict[str, str]]) -> str:
    return templates.get_template("chat_messages.html").render(messages=messages)


def _build_session_payload(runtime: ChatRuntime) -> dict[str, object]:
    manager = runtime.manager
    session_name = f"{manager.character_name} {datetime.now(tz=UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    return {
        "version": 1,
        "saved_at": datetime.now(tz=UTC).isoformat(),
        "session_name": session_name,
        "character_name": manager.character_name,
        "rag_collection": str(manager.rag_collection),
        "ui_messages": runtime.ui_messages,
        "quality": {
            "persona_anchor": {
                "character_name": manager.character_name,
                "description": manager.description,
                "scenario": manager.scenario,
                "voice_instructions": manager.voice_instructions,
            },
            "persona_drift_config": {
                "enabled": manager.runtime_config.persona_drift_enabled,
                "warning_threshold": manager.runtime_config.persona_drift_warning_threshold,
                "fail_threshold": manager.runtime_config.persona_drift_fail_threshold,
                "history_window": manager.runtime_config.persona_drift_history_window,
                "heuristic_weight": manager.runtime_config.persona_drift_heuristic_weight,
                "semantic_weight": manager.runtime_config.persona_drift_semantic_weight,
            },
        },
        "conversation_state": manager.export_conversation_state(),
    }


def _list_session_files(runtime: ChatRuntime) -> list[Path]:
    return sorted(runtime.session_dir.glob("session_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def _normalize_session_name(raw_name: str | None, fallback_name: str) -> str:
    if raw_name is None:
        return fallback_name
    candidate = " ".join(raw_name.split()).strip()
    if not candidate:
        return fallback_name
    return candidate[:80]


def _session_file_for_id(runtime: ChatRuntime, session_id: str) -> Path | None:
    if not session_id or not all(char.isalnum() or char in {"-", "_", "T", "Z"} for char in session_id):
        return None
    candidate = runtime.session_dir / f"session_{session_id}.json"
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _session_listing(runtime: ChatRuntime) -> list[dict[str, str]]:
    sessions: list[dict[str, str]] = []
    for path in _list_session_files(runtime):
        session_name = path.stem.removeprefix("session_")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_name = payload.get("session_name")
                if isinstance(raw_name, str) and raw_name.strip():
                    session_name = raw_name.strip()
        except Exception:
            logger.warning("Failed to parse session metadata from {}", path)
        sessions.append(
            {
                "session_id": path.stem.removeprefix("session_"),
                "session_name": session_name,
                "file": path.name,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
            }
        )
    return sessions


class _SessionSearchParams:
    """Parsed search parameters for _search_sessions."""

    ISO_DATE_LEN: int = 10

    def __init__(  # noqa: PLR0913
        self,
        q: str = "",
        character: str = "",
        from_date: str = "",
        to_date: str = "",
        max_results: int = 30,
        max_snippets: int = 3,
    ) -> None:
        self.q_lower = q.strip().lower()
        self.char_lower = character.strip().lower()
        self.max_results = max_results
        self.max_snippets = max_snippets
        self.from_dt = self._parse_date(from_date)
        self.to_dt = self._parse_date(to_date, end_of_day=True)

    @staticmethod
    def _parse_date(date_str: str, *, end_of_day: bool = False) -> datetime | None:
        if not date_str:
            return None
        with contextlib.suppress(ValueError):
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            return dt.replace(hour=23, minute=59, second=59) if end_of_day else dt
        return None


def _session_in_date_range(saved_at_str: str, from_dt: datetime | None, to_dt: datetime | None) -> bool:
    """Return True when the session's saved_at falls within the given range."""
    if not from_dt and not to_dt:
        return True
    saved_dt: datetime | None = None
    with contextlib.suppress(ValueError):
        saved_dt = datetime.fromisoformat(saved_at_str)
        if saved_dt.tzinfo is None:
            saved_dt = saved_dt.replace(tzinfo=UTC)
    if saved_dt is None:
        return True
    if from_dt and saved_dt < from_dt:
        return False
    return not (to_dt and saved_dt > to_dt)


def _build_snippets(ui_messages: list[dict[str, str]], q_lower: str, max_snippets: int) -> list[dict[str, str]]:
    """Extract text excerpts from messages that contain q_lower."""
    snippets: list[dict[str, str]] = []
    for msg in ui_messages:
        content = msg.get("content", "")
        idx = content.lower().find(q_lower)
        if idx >= 0:
            start = max(0, idx - 45)
            end = min(len(content), idx + len(q_lower) + 100)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(content) else ""
            snippets.append({"role": msg["role"], "excerpt": prefix + content[start:end] + suffix})
            if len(snippets) >= max_snippets:
                break
    return snippets


def _search_sessions(  # noqa: PLR0913
    runtime: ChatRuntime,
    q: str = "",
    character: str = "",
    from_date: str = "",
    to_date: str = "",
    max_results: int = 30,
    max_snippets: int = 3,
) -> list[dict[str, object]]:
    """Full-text search across saved session JSON files."""
    params = _SessionSearchParams(q, character, from_date, to_date, max_results, max_snippets)
    results: list[dict[str, object]] = []

    for path in _list_session_files(runtime):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse session file {}", path.name)
            continue
        if not isinstance(payload, dict):
            continue

        session_name = str(payload.get("session_name") or path.stem.removeprefix("session_"))
        char_name = str(payload.get("character_name") or "")
        saved_at_str = str(payload.get("saved_at") or "")
        ui_messages = _coerce_ui_messages(payload.get("ui_messages"))
        session_id = path.stem.removeprefix("session_")

        if params.char_lower and params.char_lower not in char_name.lower():
            continue
        if not _session_in_date_range(saved_at_str, params.from_dt, params.to_dt):
            continue

        snippets: list[dict[str, str]] = []
        if params.q_lower:
            name_match = params.q_lower in session_name.lower()
            snippets = _build_snippets(ui_messages, params.q_lower, params.max_snippets)
            if not name_match and not snippets:
                continue

        results.append(
            {
                "session_id": session_id,
                "session_name": session_name,
                "character_name": char_name,
                "saved_at": saved_at_str[: _SessionSearchParams.ISO_DATE_LEN],
                "snippets": snippets,
            }
        )
        if len(results) >= params.max_results:
            break

    return results


def _record_retrieval_trace(
    runtime: ChatRuntime,
    message: str,
    latency_s: float | None = None,
    chars_emitted: int | None = None,
) -> None:
    manager = runtime.manager
    turn_number = min(len(manager.user_message_history), len(manager.ai_message_history))
    trace = {
        "turn": turn_number,
        "at": datetime.now(tz=UTC).isoformat(),
        "query": message[:200],
        "latency_s": round(latency_s, 2) if latency_s is not None else None,
        "chars_emitted": chars_emitted,
        "estimated_completion_tokens": round(chars_emitted / 4) if chars_emitted is not None else None,
        "retrieval": manager.last_retrieval_debug,
        "persona": manager.last_persona_drift,
        "token_budget": dict(manager.last_token_budget),
    }
    runtime.retrieval_history.append(trace)
    history_cap = runtime.MAX_RETRIEVAL_HISTORY_ENTRIES
    if len(runtime.retrieval_history) > history_cap:
        runtime.retrieval_history = runtime.retrieval_history[-history_cap:]


def _coerce_ui_messages(raw_messages: object) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    if not isinstance(raw_messages, list):
        return parsed
    for entry in raw_messages:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        parsed.append({"role": str(role), "content": content})
    return parsed


def _character_avatar_path(character_name: str) -> Path | None:
    """Return path to the character's avatar image, or None if not found."""
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in character_name).strip("_")
    # Check character_storage first (dedicated asset folder; supports uploads)
    for ext in ("png", "jpg", "jpeg", "webp"):
        storage = Path("character_storage") / stem / f"avatar.{ext}"
        if storage.exists():
            return storage
    # Fall back to a matching image in cards/ (e.g. PNG card = its own avatar)
    for ext in ("png", "jpg", "jpeg", "webp"):
        card = Path("cards") / f"{stem}.{ext}"
        if card.exists():
            return card
    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render chat page."""
    runtime = _get_runtime(request)
    context = {
        "request": request,
        "character_name": runtime.manager.character_name,
        "messages": runtime.ui_messages,
        "model_name": str(runtime.manager.configs.get("MODEL", "")),
        "model_type": str(runtime.manager.configs.get("MODEL_TYPE", "")),
        "rag_collection": str(runtime.manager.rag_collection),
        "has_avatar": _character_avatar_path(runtime.manager.character_name) is not None,
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/characters/avatar")
async def character_avatar(request: Request) -> FileResponse:
    """Serve the current character's avatar image."""
    runtime = _get_runtime(request)
    path = _character_avatar_path(runtime.manager.character_name)
    if path is None:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=404, detail="No avatar found")
    return FileResponse(path)


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Lightweight health endpoint for runtime diagnostics."""
    runtime = _get_runtime(request)
    return {
        "status": "ok",
        "character": runtime.manager.character_name,
        "rag_collection": str(runtime.manager.rag_collection),
        "busy": runtime.lock.locked(),
    }


@app.get("/healthz/full")
async def health_full(request: Request) -> dict[str, object]:
    """Extended health endpoint with non-sensitive runtime diagnostics."""
    runtime = _get_runtime(request)
    manager = runtime.manager
    now = time.time()
    model_path = str(manager.configs.get("MODEL", ""))

    return {
        "status": "ok",
        "runtime": {
            "busy": runtime.lock.locked(),
            "started_at": runtime.started_at,
            "uptime_seconds": round(now - runtime.started_at, 3),
        },
        "character": {
            "name": manager.character_name,
            "rag_collection": str(manager.rag_collection),
            "has_first_message": bool(manager.first_message),
        },
        "model": {
            "type": str(manager.configs.get("MODEL_TYPE", "")),
            "path": model_path,
            "model_file": Path(model_path).name if model_path else "",
            "n_ctx_config": int(manager.configs.get("N_CTX", 0) or 0),
            "kv_cache_quant": str(manager.configs.get("KV_CACHE_QUANT", "")),
            "gpu_layers": str(manager.configs.get("LAYERS", "")),
        },
        "retrieval": {
            "rag_k": int(manager.configs.get("RAG_K", 0) or 0),
            "rag_k_mes": int(manager.configs.get("RAG_K_MES", 0) or 0),
            "use_mmr": bool(manager.configs.get("USE_MMR", False)),
            "use_dynamic_context": bool(manager.configs.get("USE_DYNAMIC_CONTEXT", False)),
            "max_history_turns": int(manager.configs.get("MAX_HISTORY_TURNS", 0) or 0),
            "history_turns": min(len(manager.user_message_history), len(manager.ai_message_history)),
        },
    }


@app.get("/chat/debug")
async def chat_debug(request: Request) -> dict[str, object]:
    """Expose retrieval debug metadata from the latest turn."""
    runtime = _get_runtime(request)
    return {
        "status": "ok",
        "retrieval": runtime.manager.last_retrieval_debug,
        "persona": {
            "last": runtime.manager.last_persona_drift,
            "history": list(runtime.manager.persona_drift_history),
            "summary": runtime.manager.get_persona_drift_summary(),
        },
        "history_turns": min(len(runtime.manager.user_message_history), len(runtime.manager.ai_message_history)),
    }


@app.get("/chat/debug/history")
async def chat_debug_history(request: Request) -> dict[str, object]:
    """Expose retrieval debug history keyed by turn."""
    runtime = _get_runtime(request)
    return {
        "status": "ok",
        "count": len(runtime.retrieval_history),
        "history": runtime.retrieval_history,
    }


@app.get("/chat/diagnostics", response_class=HTMLResponse)
async def chat_diagnostics(request: Request) -> HTMLResponse:
    """Render per-turn diagnostics table as an HTMX partial."""
    runtime = _get_runtime(request)
    manager = runtime.manager
    history = list(reversed(runtime.retrieval_history))
    html = templates.get_template("diagnostics_panel.html").render(
        history=history,
        last_budget=dict(manager.last_token_budget),
        warn_threshold=manager.runtime_config.persona_drift_warning_threshold,
        fail_threshold=manager.runtime_config.persona_drift_fail_threshold,
    )
    return HTMLResponse(content=html)


@app.get("/chat/export/bundle")
async def chat_export_bundle(request: Request) -> Response:
    """Download a ZIP bundle containing conversation, retrieval traces, and drift history."""
    runtime = _get_runtime(request)
    manager = runtime.manager
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    turn_count = min(len(manager.user_message_history), len(manager.ai_message_history))
    manifest = {
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "character_name": manager.character_name,
        "rag_collection": str(manager.rag_collection),
        "turns": turn_count,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr(
            "conversation.json",
            json.dumps(_build_session_payload(runtime), ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "retrieval_traces.json",
            json.dumps(runtime.retrieval_history, ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "drift_history.json",
            json.dumps(list(manager.persona_drift_trace), ensure_ascii=False, indent=2),
        )
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=bundle_{ts}.zip"},
    )


@app.post("/chat/send", response_class=HTMLResponse)
async def chat_send(request: Request, message: str = Form(...)) -> HTMLResponse:
    """Append user message and assistant placeholder via HTMX."""
    cleaned_message = message.strip()
    if not cleaned_message:
        return HTMLResponse(content="", status_code=204)
    runtime = _get_runtime(request)
    logger.debug("chat_send received message_chars={}", len(cleaned_message))
    runtime.ui_messages.append({"role": "user", "content": cleaned_message})

    assistant_id = f"assistant-{uuid.uuid4().hex}"
    html = templates.get_template("chat_message_pair.html").render(
        user_message=cleaned_message,
        assistant_id=assistant_id,
        stream_message=cleaned_message,
    )
    return HTMLResponse(content=html)


def _ask_question_worker(message: str, stream_callback: Callable[[str], None], manager: ConversationManager) -> None:
    """Run ask_question in a dedicated thread event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        first_token_event = threading.Event()
        loop.run_until_complete(manager.ask_question(message, first_token_event, stream_callback))
    finally:
        loop.close()


async def _stream_answer(message: str, manager: ConversationManager) -> AsyncGenerator[str]:
    """Generate token stream from conversation manager."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def stream_callback(chunk: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, chunk)

    ask_task = asyncio.create_task(asyncio.to_thread(_ask_question_worker, message, stream_callback, manager))

    try:
        while True:
            if ask_task.done() and queue.empty():
                break
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            yield chunk

        await ask_task
    except Exception as error:
        logger.exception("Error while streaming web response")
        yield f"\n[Error: {error!s}]"
    finally:
        if not ask_task.done():
            ask_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ask_task


@app.post("/chat/stream")
async def chat_stream(request: Request, payload: StreamRequest) -> StreamingResponse:
    """Stream assistant tokens as plain text chunks."""
    runtime = _get_runtime(request)
    logger.debug("chat_stream start message_chars={}", len(payload.message))

    async def guarded_stream() -> AsyncGenerator[str]:
        async with runtime.lock:
            start = time.monotonic()
            emitted_chunks = 0
            emitted_chars = 0
            prior_turn_count = min(len(runtime.manager.user_message_history), len(runtime.manager.ai_message_history))
            async for chunk in _stream_answer(payload.message, runtime.manager):
                emitted_chunks += 1
                emitted_chars += len(chunk)
                yield chunk
            final_turn_count = min(len(runtime.manager.user_message_history), len(runtime.manager.ai_message_history))
            if final_turn_count > prior_turn_count and runtime.manager.ai_message_history:
                latest_assistant = list(runtime.manager.ai_message_history)[-1]
                if payload.continue_mode:
                    for item in reversed(runtime.ui_messages):
                        if item.get("role") != "assistant":
                            continue
                        prior_content = item.get("content", "")
                        if isinstance(prior_content, str):
                            item["content"] = f"{prior_content}{latest_assistant}"
                            break
                    else:
                        runtime.ui_messages.append({"role": "assistant", "content": latest_assistant})
                else:
                    runtime.ui_messages.append({"role": "assistant", "content": latest_assistant})
                    elapsed = time.monotonic() - start
                    _record_retrieval_trace(runtime, payload.message, latency_s=elapsed, chars_emitted=emitted_chars)
            elapsed = time.monotonic() - start
            logger.debug(
                "chat_stream done chunks={} chars={} elapsed={:.2f}s",
                emitted_chunks,
                emitted_chars,
                elapsed,
            )

    return StreamingResponse(guarded_stream(), media_type="text/plain; charset=utf-8")


@app.post("/chat/action/help", response_class=HTMLResponse)
async def chat_action_help(request: Request) -> HTMLResponse:
    """Append an in-chat help message."""
    runtime = _get_runtime(request)
    help_message = (
        "Quick actions: Clear, Reload, Continue, Save Session, Load Latest, Copy Last, Export TXT, Export JSON.\n"
        "Keyboard: Ctrl+Enter send, ArrowUp/ArrowDown prompt history, Alt+H help, Alt+C clear, "
        "Alt+R reload, Alt+N continue."
    )
    runtime.ui_messages.append({"role": "assistant", "content": help_message})
    html = templates.get_template("chat_single_message.html").render(role="assistant", content=help_message)
    return HTMLResponse(content=html)


@app.post("/chat/action/clear", response_class=HTMLResponse)
async def chat_action_clear(request: Request) -> HTMLResponse:
    """Clear chat history while keeping current model/runtime."""
    runtime = _get_runtime(request)
    async with runtime.lock:
        runtime.manager.clear_conversation_state()
        runtime.reset_ui_messages()
        runtime.retrieval_history = []
        return HTMLResponse(content=_render_chat_log(runtime.ui_messages))


@app.post("/chat/action/continue")
async def chat_action_continue(request: Request) -> JSONResponse:
    """Return a continuation prompt for seamless in-place assistant extension."""
    runtime = _get_runtime(request)
    if not runtime.manager.ai_message_history:
        return Response(status_code=204)
    return JSONResponse(content={"status": "ok", "message": CONTINUE_PROMPT})


@app.post("/chat/action/reload", response_class=HTMLResponse)
async def chat_action_reload(request: Request) -> HTMLResponse:
    """Recreate ConversationManager and reset web-visible history."""
    runtime = _get_runtime(request)
    async with runtime.lock:
        runtime.manager = await asyncio.to_thread(ConversationManager)
        runtime.started_at = time.time()
        runtime.reset_ui_messages()
        runtime.retrieval_history = []
        return HTMLResponse(content=_render_chat_log(runtime.ui_messages))


@app.post("/chat/session/save")
async def chat_session_save(request: Request, session_name: str | None = Form(default=None)) -> JSONResponse:
    """Persist current web chat session to disk."""
    runtime = _get_runtime(request)
    async with runtime.lock:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        session_path = runtime.session_dir / f"session_{session_id}.json"
        payload = _build_session_payload(runtime)
        provided_name = str(session_name) if session_name is not None else None
        payload["session_name"] = _normalize_session_name(provided_name, str(payload["session_name"]))
        session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return JSONResponse(
            content={
                "status": "ok",
                "session_id": session_id,
                "session_name": payload["session_name"],
                "path": str(session_path),
            }
        )


@app.get("/chat/session/list")
async def chat_session_list(request: Request) -> dict[str, object]:
    """List available persisted web sessions."""
    runtime = _get_runtime(request)
    return {
        "status": "ok",
        "sessions": _session_listing(runtime),
    }


@app.post("/chat/session/load-latest", response_class=HTMLResponse)
async def chat_session_load_latest(request: Request) -> HTMLResponse:
    """Load the most recently saved web chat session."""
    runtime = _get_runtime(request)
    async with runtime.lock:
        files = _list_session_files(runtime)
        if not files:
            return HTMLResponse(content="", status_code=204)

        latest = files[0]
        raw = json.loads(latest.read_text(encoding="utf-8"))
        state = raw.get("conversation_state", {}) if isinstance(raw, dict) else {}
        ui_messages = _coerce_ui_messages(raw.get("ui_messages") if isinstance(raw, dict) else None)
        runtime.manager.import_conversation_state(state if isinstance(state, dict) else {})
        manager = runtime.manager

        runtime.ui_messages = ui_messages
        if not runtime.ui_messages and manager.first_message:
            runtime.ui_messages = [{"role": "assistant", "content": manager.first_message}]

        runtime.retrieval_history = []

        return HTMLResponse(content=_render_chat_log(runtime.ui_messages))


@app.post("/chat/session/load", response_class=HTMLResponse)
async def chat_session_load(request: Request, session_id: str = Form(...)) -> HTMLResponse:
    """Load a selected web chat session by ID."""
    runtime = _get_runtime(request)
    async with runtime.lock:
        session_file = _session_file_for_id(runtime, session_id.strip())
        if session_file is None:
            return HTMLResponse(content="", status_code=404)

        raw = json.loads(session_file.read_text(encoding="utf-8"))
        state = raw.get("conversation_state", {}) if isinstance(raw, dict) else {}
        ui_messages = _coerce_ui_messages(raw.get("ui_messages") if isinstance(raw, dict) else None)
        runtime.manager.import_conversation_state(state if isinstance(state, dict) else {})
        manager = runtime.manager
        runtime.ui_messages = ui_messages
        if not runtime.ui_messages and manager.first_message:
            runtime.ui_messages = [{"role": "assistant", "content": manager.first_message}]
        runtime.retrieval_history = []
        return HTMLResponse(content=_render_chat_log(runtime.ui_messages))


@app.get("/sessions/search", response_class=HTMLResponse)
async def sessions_search(
    request: Request,
    q: str = Query(default=""),
    character: str = Query(default=""),
    from_date: str = Query(default=""),
    to_date: str = Query(default=""),
) -> HTMLResponse:
    """Full-text search across saved session JSON files; returns an HTMX partial."""
    if not q.strip() and not character.strip() and not from_date and not to_date:
        return HTMLResponse(content="")
    runtime = _get_runtime(request)
    results = await asyncio.to_thread(_search_sessions, runtime, q, character, from_date, to_date)
    return templates.TemplateResponse(
        "sessions_search_results.html",
        {"request": request, "results": results, "q": q},
    )


def _render_profiles_panel(request: Request) -> HTMLResponse:
    runtime = _get_runtime(request)
    store = _get_profile_store(request)
    rendered_html = templates.get_template("presets_panel.html").render(
        profiles=store.list_profiles(),
        current=store.current_values(runtime.manager.runtime_config),
    )
    return HTMLResponse(content=rendered_html)


@app.get("/settings/profiles", response_class=HTMLResponse)
async def settings_profiles_list(request: Request) -> HTMLResponse:
    """Render the presets panel partial."""
    return _render_profiles_panel(request)


@app.post("/settings/profiles/save", response_class=HTMLResponse)
async def settings_profiles_save(request: Request, name: str = Form(...)) -> HTMLResponse:
    """Save current runtime config as a named profile, then re-render the panel."""
    runtime = _get_runtime(request)
    store = _get_profile_store(request)
    clean_name = name.strip()[:40]
    if clean_name:
        store.save_profile(clean_name, runtime.manager.runtime_config)
    return _render_profiles_panel(request)


@app.post("/settings/profiles/apply", response_class=HTMLResponse)
async def settings_profiles_apply(request: Request, name: str = Form(...)) -> HTMLResponse:
    """Apply a saved profile to the live runtime config, then re-render the panel."""
    runtime = _get_runtime(request)
    store = _get_profile_store(request)
    async with runtime.lock:
        try:
            new_config, changed = store.apply_profile(name, runtime.manager.runtime_config)
            runtime.manager.runtime_config = new_config
            runtime.manager.rag_k = new_config.rag_k
            runtime.manager.rag_k_mes = new_config.rag_k_mes
            logger.info("Applied profile {!r}; changed fields: {}", name, changed)
        except KeyError:
            logger.warning("Profile {!r} not found", name)
        return _render_profiles_panel(request)


@app.post("/settings/profiles/delete", response_class=HTMLResponse)
async def settings_profiles_delete(request: Request, name: str = Form(...)) -> HTMLResponse:
    """Delete a saved profile, then re-render the panel."""
    store = _get_profile_store(request)
    store.delete_profile(name)
    return _render_profiles_panel(request)


# ---------------------------------------------------------------------------
# RAG Management UI
# ---------------------------------------------------------------------------


@app.get("/rag", response_class=HTMLResponse)
async def rag_page(request: Request) -> HTMLResponse:
    """Render the standalone RAG management page."""
    return templates.TemplateResponse("rag/layout.html", {"request": request})


@app.get("/rag/collections", response_class=HTMLResponse)
async def rag_collections(request: Request) -> HTMLResponse:
    config = _get_rag_config(request)
    collections = await asyncio.to_thread(rag_manager.list_collections, config)
    stems = await asyncio.to_thread(rag_manager.list_rag_stems, config)
    return templates.TemplateResponse(
        "rag/collections_list.html",
        {"request": request, "collections": collections, "existing_stems": stems},
    )


@app.get("/rag/collections/{name}", response_class=HTMLResponse)
async def rag_collection_detail(request: Request, name: str) -> HTMLResponse:
    config = _get_rag_config(request)
    info = await asyncio.to_thread(rag_manager.collection_info, config, name)
    if info is None:
        return HTMLResponse(content=f"<p>Collection <b>{html.escape(name)}</b> not found.</p>", status_code=404)
    return templates.TemplateResponse(
        "rag/collection_detail.html",
        {"request": request, "info": info},
    )


@app.delete("/rag/collections/{name}", response_class=HTMLResponse)
async def rag_collection_delete(request: Request, name: str) -> HTMLResponse:
    config = _get_rag_config(request)
    try:
        await asyncio.to_thread(rag_manager.delete_collection, config, name)
    except Exception as exc:
        logger.warning("Failed to delete collection {!r}: {}", name, exc)
        return HTMLResponse(
            content=f"<p class='text-error'>Could not delete collection <b>{html.escape(name)}</b>.</p>",
            status_code=404,
        )
    collections = await asyncio.to_thread(rag_manager.list_collections, config)
    return templates.TemplateResponse(
        "rag/collections_list.html",
        {"request": request, "collections": collections, "deleted": name},
    )


@app.post("/rag/collections/{name}/query", response_class=HTMLResponse)
async def rag_collection_query(
    request: Request,
    name: str,
    query: str = Form(...),
    k: int = Form(5),
) -> HTMLResponse:
    config = _get_rag_config(request)
    try:
        results = await asyncio.to_thread(rag_manager.query_collection, config, name, query, k)
    except Exception as exc:
        logger.warning("Query failed for collection {!r}: {}", name, exc)
        return HTMLResponse(
            content="<p class='text-error'>Query failed. See server logs for details.</p>",
            status_code=400,
        )
    return templates.TemplateResponse(
        "rag/query_results.html",
        {"request": request, "results": results, "query": query, "collection": name},
    )


@app.post("/rag/collections/{name}/push", response_class=HTMLResponse)
async def rag_collection_push(
    request: Request,
    name: str,
    stem: str = Form(...),
) -> HTMLResponse:
    clean_stem = stem.strip()
    if not rag_manager.is_valid_stem(clean_stem):
        _msg = "Invalid stem: only letters, digits, underscores, and hyphens are allowed."
        return HTMLResponse(content=f"<p class='text-error'>{_msg}</p>", status_code=400)
    config = _get_rag_config(request)
    job_store = _get_job_store(request)
    job_id = job_store.submit(rag_manager.push_collection, config, clean_stem, name)
    return templates.TemplateResponse(
        "rag/push_status.html",
        {
            "request": request,
            "job_id": job_id,
            "status": "pending",
            "elapsed_s": 0,
            "result": None,
            "error": None,
            "kind": "push",
        },
    )


@app.post("/rag/collections/{name}/backfill-fingerprint", response_class=HTMLResponse)
async def rag_collection_backfill(request: Request, name: str) -> HTMLResponse:
    config = _get_rag_config(request)
    backfilled = await asyncio.to_thread(rag_manager.backfill_fingerprint, config, name)
    info = await asyncio.to_thread(rag_manager.collection_info, config, name)
    return templates.TemplateResponse(
        "rag/collection_detail.html",
        {"request": request, "info": info, "backfilled": backfilled},
    )


@app.get("/rag/files", response_class=HTMLResponse)
async def rag_files(request: Request) -> HTMLResponse:
    config = _get_rag_config(request)
    files = await asyncio.to_thread(rag_manager.list_rag_files, config)
    return templates.TemplateResponse(
        "rag/files_list.html",
        {"request": request, "files": files},
    )


@app.get("/rag/files/{filename}", response_class=HTMLResponse)
async def rag_file_view(request: Request, filename: str) -> HTMLResponse:
    config = _get_rag_config(request)
    content = await asyncio.to_thread(rag_manager.file_content, config, filename)
    if content is None:
        return HTMLResponse(
            content=f"<p>File <b>{html.escape(filename)}</b> not found or not accessible.</p>",
            status_code=404,
        )
    return templates.TemplateResponse(
        "rag/file_view.html",
        {"request": request, "filename": filename, "content": content},
    )


def _validate_upload_content(content: bytes) -> HTMLResponse | None:
    """Validate uploaded file content; return an error HTMLResponse or None if valid."""
    if len(content) > _MAX_UPLOAD_BYTES:
        return HTMLResponse(
            content=(
                f"<p class='text-error'>File too large:"
                f" maximum allowed size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.</p>"
            ),
            status_code=400,
        )
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return HTMLResponse(
            content="<p class='text-error'>Invalid file: only UTF-8 encoded text files are accepted.</p>",
            status_code=400,
        )
    return None


@app.post("/rag/files/upload", response_class=HTMLResponse)
async def rag_file_upload(
    request: Request,
    stem: str = Form(...),
    file: UploadFile = File(...),  # noqa: B008
    collection_name: str = Form(""),
) -> HTMLResponse:
    """Upload a .txt source file to rag_data/. Optionally create a collection immediately."""
    clean_stem = stem.strip()
    if not rag_manager.is_valid_stem(clean_stem):
        return HTMLResponse(content=_INVALID_STEM_HTML, status_code=400)
    content = await file.read()
    if (err := _validate_upload_content(content)) is not None:
        return err
    config = _get_rag_config(request)
    try:
        file_info = await asyncio.to_thread(rag_manager.save_rag_file, config, clean_stem, content)
    except Exception as exc:
        logger.warning("Failed to save uploaded file {!r}: {}", clean_stem, exc)
        return HTMLResponse(
            content=f"<p class='text-error'>Upload failed: {html.escape(str(exc))}</p>",
            status_code=500,
        )
    clean_coll = collection_name.strip()
    if clean_coll:
        if not rag_manager.is_valid_stem(clean_coll):
            return HTMLResponse(content=_INVALID_COLL_HTML, status_code=400)
        job_store = _get_job_store(request)
        job_id = job_store.submit(rag_manager.push_collection, config, clean_stem, clean_coll)
        return templates.TemplateResponse(
            "rag/upload_result.html",
            {
                "request": request,
                "filename": file_info["name"],
                "collection_name": clean_coll,
                "job_id": job_id,
                "status": "pending",
                "elapsed_s": 0,
                "result": None,
                "error": None,
                "kind": "push",
            },
        )
    files = await asyncio.to_thread(rag_manager.list_rag_files, config)
    return templates.TemplateResponse(
        "rag/files_list.html",
        {"request": request, "files": files, "uploaded": file_info["name"]},
    )


@app.post("/rag/collections", response_class=HTMLResponse)
async def rag_collection_create(
    request: Request,
    collection_name: str = Form(...),
    stem: str = Form(...),
) -> HTMLResponse:
    """Create a new collection by ingesting an existing rag_data source file."""
    clean_stem = stem.strip()
    clean_name = collection_name.strip()
    if not rag_manager.is_valid_stem(clean_stem):
        return HTMLResponse(content=_INVALID_STEM_HTML, status_code=400)
    if not clean_name:
        return HTMLResponse(
            content="<p class='text-error'>Collection name is required.</p>",
            status_code=400,
        )
    config = _get_rag_config(request)
    job_store = _get_job_store(request)
    job_id = job_store.submit(rag_manager.push_collection, config, clean_stem, clean_name)
    return templates.TemplateResponse(
        "rag/push_status.html",
        {
            "request": request,
            "job_id": job_id,
            "status": "pending",
            "elapsed_s": 0,
            "result": None,
            "error": None,
            "kind": "push",
        },
    )


@app.post("/rag/lint", response_class=HTMLResponse)
async def rag_lint(request: Request) -> HTMLResponse:
    config = _get_rag_config(request)
    reports = await asyncio.to_thread(rag_manager.run_lint, config)
    return templates.TemplateResponse(
        "rag/lint_results.html",
        {"request": request, "reports": reports, "fixed": False},
    )


@app.post("/rag/lint/fix", response_class=HTMLResponse)
async def rag_lint_fix(request: Request) -> HTMLResponse:
    config = _get_rag_config(request)
    reports = await asyncio.to_thread(rag_manager.run_lint, config, auto_fix=True)
    return templates.TemplateResponse(
        "rag/lint_results.html",
        {"request": request, "reports": reports, "fixed": True},
    )


@app.post("/rag/coverage", response_class=HTMLResponse)
async def rag_coverage(request: Request, stem: str = Form(...)) -> HTMLResponse:
    clean_stem = stem.strip()
    if not rag_manager.is_valid_stem(clean_stem):
        _msg = "Invalid stem: only letters, digits, underscores, and hyphens are allowed."
        return HTMLResponse(content=f"<p class='text-error'>{_msg}</p>", status_code=400)
    config = _get_rag_config(request)
    result = await asyncio.to_thread(rag_manager.run_coverage, config, clean_stem)
    if result is None:
        return HTMLResponse(
            content="<p>Coverage data not found (missing .txt or .json pair).</p>",
            status_code=404,
        )
    return templates.TemplateResponse(
        "rag/coverage_report.html",
        {"request": request, "result": result},
    )


@app.get("/rag/evaluate", response_class=HTMLResponse)
async def rag_evaluate_index(request: Request) -> HTMLResponse:
    fixture_packs = await asyncio.to_thread(rag_manager.list_fixture_packs)
    trends = await asyncio.to_thread(rag_manager.get_fixture_trends)
    return templates.TemplateResponse(
        "rag/evaluate_index.html",
        {"request": request, "fixture_packs": fixture_packs, "trends": trends},
    )


@app.post("/rag/evaluate/run", response_class=HTMLResponse)
async def rag_evaluate_run(
    request: Request,
    fixture_file: str = Form(...),
) -> HTMLResponse:
    available = await asyncio.to_thread(rag_manager.list_fixture_packs)
    if fixture_file not in available:
        return HTMLResponse(
            content="<p class='text-error'>Unknown fixture pack.</p>",
            status_code=400,
        )
    config = _get_rag_config(request)
    job_store = _get_job_store(request)
    job_id = job_store.submit(rag_manager.run_evaluate_fixtures, config, fixture_file)
    return templates.TemplateResponse(
        "rag/push_status.html",
        {
            "request": request,
            "job_id": job_id,
            "status": "pending",
            "elapsed_s": 0,
            "result": None,
            "error": None,
            "kind": "evaluate",
            "label": "Evaluating fixtures…",
        },
    )


@app.get("/rag/evaluate/trends", response_class=HTMLResponse)
async def rag_evaluate_trends(request: Request) -> HTMLResponse:
    trends = await asyncio.to_thread(rag_manager.get_fixture_trends)
    return templates.TemplateResponse(
        "rag/trends_table.html",
        {"request": request, "trends": trends},
    )


@app.get("/rag/benchmark", response_class=HTMLResponse)
async def rag_benchmark(request: Request) -> HTMLResponse:
    results = await asyncio.to_thread(rag_manager.get_benchmark_results)
    return templates.TemplateResponse(
        "rag/benchmark_results.html",
        {"request": request, "results": results},
    )


@app.get("/rag/jobs/{job_id}", response_class=HTMLResponse)
async def rag_job_status(
    request: Request,
    job_id: str,
    kind: str = Query("push"),
) -> HTMLResponse:
    job_store = _get_job_store(request)
    job = job_store.get(job_id)
    if job is None:
        return HTMLResponse('<span class="badge badge-error">Job not found</span>')
    if job["status"] == "done" and kind == "evaluate":
        return templates.TemplateResponse(
            "rag/evaluate_results.html",
            {"request": request, "result": job["result"]},
        )
    return templates.TemplateResponse(
        "rag/push_status.html",
        {
            "request": request,
            "job_id": job_id,
            "status": job["status"],
            "elapsed_s": round(job["elapsed_s"], 1),
            "result": job["result"],
            "error": job["error"],
            "kind": kind,
        },
    )
