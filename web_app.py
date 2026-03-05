"""FastAPI + Jinja2 + HTMX web chat interface for light-chat."""

import asyncio
import contextlib
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel

from core.config import configure_logging, load_app_config
from core.conversation_manager import ConversationManager


class StreamRequest(BaseModel):
    """Streaming request payload."""

    message: str


class ChatRuntime:
    """Holds shared chat runtime state for web requests."""

    def __init__(self, manager: ConversationManager) -> None:
        self.manager = manager
        self.lock = asyncio.Lock()
        self.started_at = time.time()


templates = Jinja2Templates(directory="templates")


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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render chat page."""
    runtime = _get_runtime(request)
    context = {
        "request": request,
        "character_name": runtime.manager.character_name,
        "first_message": runtime.manager.first_message,
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


@app.post("/chat/send", response_class=HTMLResponse)
async def chat_send(message: str = Form(...)) -> HTMLResponse:
    """Append user message and assistant placeholder via HTMX."""
    cleaned_message = message.strip()
    if not cleaned_message:
        return HTMLResponse(content="", status_code=204)
    logger.debug("chat_send received message_chars={}", len(cleaned_message))

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
            async for chunk in _stream_answer(payload.message, runtime.manager):
                emitted_chunks += 1
                emitted_chars += len(chunk)
                yield chunk
            elapsed = time.monotonic() - start
            logger.debug(
                "chat_stream done chunks={} chars={} elapsed={:.2f}s",
                emitted_chunks,
                emitted_chars,
                elapsed,
            )

    return StreamingResponse(guarded_stream(), media_type="text/plain; charset=utf-8")
