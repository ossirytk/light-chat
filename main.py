import asyncio
import contextlib
import json
import logging
import sys
import threading
import time
from itertools import cycle
from pathlib import Path

from loguru import logger

from core.conversation_manager import ConversationManager


def load_app_config() -> dict:
    config_path = Path("./configs/") / "appconf.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def configure_logging(app_config: dict) -> None:
    show_logs = bool(app_config.get("SHOW_LOGS", True))
    log_level = str(app_config.get("LOG_LEVEL", "DEBUG")).upper()
    log_to_file = bool(app_config.get("LOG_TO_FILE", True))
    log_file = str(app_config.get("LOG_FILE", "./logs/light-chat.log"))

    logger.remove()

    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level=log_level, rotation="10 MB", retention=5)

    if show_logs:
        logging.basicConfig(level=log_level)
        logger.add(sys.stderr, level=log_level)
    else:
        logging.disable(logging.CRITICAL)


def run_spinner(message: str, stop_event: threading.Event) -> None:
    start = time.monotonic()
    for ch in cycle("-\\|/"):
        if stop_event.is_set():
            break
        elapsed = time.monotonic() - start
        line = f"{message} {ch} {elapsed:0.1f}s"
        sys.stderr.write(f"\r{line}")
        sys.stderr.flush()
        time.sleep(0.1)
    clear_len = len(message) + 8
    sys.stderr.write(f"\r{' ' * clear_len}\r")
    sys.stderr.flush()


async def main() -> None:
    """Main interactive chat loop."""
    app_config = load_app_config()
    configure_logging(app_config)
    spinner_stop = threading.Event()
    spinner_thread = threading.Thread(
        target=run_spinner,
        args=("Loading model", spinner_stop),
        daemon=True,
    )
    spinner_thread.start()
    conversation_manager = ConversationManager()
    spinner_stop.set()
    spinner_thread.join()
    if conversation_manager.first_message:
        print(f"{conversation_manager.first_message}")  # noqa: T201
        print()  # noqa: T201
    try:
        while True:
            try:
                query = input("User: ").strip()
                if not query:
                    continue
                thinking_stop = threading.Event()
                thinking_thread = threading.Thread(
                    target=run_spinner,
                    args=("Thinking", thinking_stop),
                    daemon=True,
                )
                thinking_thread.start()

                spinner_stopped = False

                def stop_spinner_once(
                    stop_event: threading.Event = thinking_stop,
                    spinner: threading.Thread = thinking_thread,
                ) -> None:
                    nonlocal spinner_stopped
                    if spinner_stopped:
                        return
                    stop_event.set()
                    spinner.join()
                    spinner_stopped = True

                def stream_callback(chunk: str) -> None:
                    stop_spinner_once()
                    print(chunk, flush=True, end="")  # noqa: T201

                await conversation_manager.ask_question(query, stream_callback=stream_callback)
                stop_spinner_once()
                print()  # noqa: T201
            except KeyboardInterrupt:
                print()  # noqa: T201
                raise
    finally:
        with contextlib.suppress(Exception):
            spinner_stop.set()
            spinner_thread.join()
        del conversation_manager


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.debug("Caught keyboard interrupt. Exiting...")
