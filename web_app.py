"""FastAPI + Jinja2 + HTMX web chat interface for light-chat."""

import asyncio
import contextlib
import json
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel

from core.config import configure_logging, load_app_config
from core.conversation_manager import ConversationManager


class StreamRequest(BaseModel):
    """Streaming request payload."""

    message: str
    continue_mode: bool = False


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
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            del app.state.chat_runtime


app = FastAPI(title="light-chat web", lifespan=lifespan)


def _get_runtime(request: Request) -> ChatRuntime:
    return request.app.state.chat_runtime


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


def _record_retrieval_trace(runtime: ChatRuntime, message: str) -> None:
    manager = runtime.manager
    turn_number = min(len(manager.user_message_history), len(manager.ai_message_history))
    trace = {
        "turn": turn_number,
        "at": datetime.now(tz=UTC).isoformat(),
        "query": message[:200],
        "retrieval": manager.last_retrieval_debug,
        "persona": manager.last_persona_drift,
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
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


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
                    _record_retrieval_trace(runtime, payload.message)
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
