import asyncio
import json
import logging
import sys
import threading
import time
from itertools import cycle
from pathlib import Path

from loguru import logger

from conversation_manager import ConversationManager


def load_app_config() -> dict:
    config_path = Path("./configs/") / "appconf.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def configure_logging(app_config: dict) -> None:
    show_logs = bool(app_config.get("SHOW_LOGS", True))
    log_level = str(app_config.get("LOG_LEVEL", "DEBUG")).upper()
    if show_logs:
        logging.basicConfig(level=log_level)
        logger.remove()
        logger.add(sys.stderr, level=log_level)
    else:
        logging.disable(logging.CRITICAL)
        logger.remove()


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
                query = input("User: ")
                thinking_stop = threading.Event()
                thinking_thread = threading.Thread(
                    target=run_spinner,
                    args=("Thinking", thinking_stop),
                    daemon=True,
                )
                thinking_thread.start()
                await conversation_manager.ask_question(query, thinking_stop)
                thinking_stop.set()
                thinking_thread.join()
                print()  # noqa: T201
            except KeyboardInterrupt:
                print()  # noqa: T201
                raise
    finally:
        del conversation_manager


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.debug("Caught keyboard interrupt. Exiting...")

