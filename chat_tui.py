"""Modern TUI interface for light-chat using Textual."""

import asyncio
import json
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from loguru import logger
from rich.panel import Panel
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.widgets import Footer, Header, Input, Static

from core.conversation_manager import ConversationManager


def load_app_config() -> dict:
    """Load application configuration."""
    config_path = Path("./configs/") / "appconf.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def configure_logging(app_config: dict) -> None:
    """Configure logging based on app config."""
    show_logs = bool(app_config.get("SHOW_LOGS", True))
    log_level = str(app_config.get("LOG_LEVEL", "DEBUG")).upper()
    if show_logs:
        logging.basicConfig(level=log_level)
        logger.remove()
        logger.add(sys.stderr, level=log_level)
    else:
        logging.disable(logging.CRITICAL)
        logger.remove()


class CharacterCard(Static):
    """Widget to display character information in a styled panel."""

    def __init__(
        self,
        character_name: str = "",
        model_name: str = "",
        model_type: str = "",
        rag_collection: str = "",
        summary: str = "",
    ) -> None:
        """Initialize character card widget."""
        super().__init__()
        self.character_name = character_name
        self.model_name = model_name
        self.model_type = model_type
        self.rag_collection = rag_collection
        self.summary = summary
        self.show_summary = False

    def render(self) -> Panel:
        """Render the character card panel."""
        if not self.character_name:
            content = Text("Loading character...", style="italic dim")
        else:
            content = Text()
            content.append(f"{self.character_name}\n", style="bold cyan")
            content.append("─" * 30 + "\n", style="dim")
            if self.show_summary:
                content.append("Summary\n", style="bold")
                content.append(f"{self.summary or 'No summary available.'}\n\n", style="white")
                content.append("[F2] Switch to metadata", style="dim")
            else:
                content.append("Model: ", style="bold")
                content.append(f"{self.model_name or 'N/A'}\n", style="white")
                content.append("Type: ", style="bold")
                content.append(f"{self.model_type or 'N/A'}\n", style="white")
                content.append("RAG Collection: ", style="bold")
                content.append(f"{self.rag_collection or 'N/A'}\n\n", style="white")
                content.append("[F2] Switch to summary", style="dim")

        return Panel(
            content,
            title="[bold magenta]Character[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )

    def update_character(
        self,
        name: str,
        model_name: str,
        model_type: str,
        rag_collection: str,
        summary: str,
    ) -> None:
        """Update character and model information and refresh display."""
        self.character_name = name
        self.model_name = model_name
        self.model_type = model_type
        self.rag_collection = rag_collection
        self.summary = summary
        self.refresh()

    def toggle_mode(self) -> None:
        """Toggle between metadata and summary display modes."""
        self.show_summary = not self.show_summary
        self.refresh()


class ChatMessage(Static):
    """Widget for displaying a single chat message."""

    def __init__(self, sender: str, message: str, *, is_user: bool = False) -> None:
        """Initialize chat message widget."""
        super().__init__()
        self.sender = sender
        self.message = message
        self.is_user = is_user
        self._rendered_text = self._build_message_text()

    def _build_message_text(self) -> Text:
        """Build the message text with styling."""
        if self.is_user:
            style = "bold green"
            prefix = "▶"
        else:
            style = "bold cyan"
            prefix = "◀"

        text = Text()
        text.append(f"{prefix} {self.sender}: ", style=style)
        text.append(self.message, style="white")
        return text

    def compose(self) -> ComposeResult:
        """Compose the chat message."""
        yield Static(self._rendered_text)

    def append_text(self, chunk: str) -> None:
        """Append text to the message (for streaming)."""
        self.message += chunk
        self._rendered_text = self._build_message_text()
        # Update the child Static widget
        if self.children:
            self.children[0].update(self._rendered_text)


class ChatLog(VerticalScroll):
    """Scrollable chat log container."""

    def add_message(self, sender: str, message: str, *, is_user: bool = False) -> None:
        """Add a message to the chat log."""
        msg_widget = ChatMessage(sender, message, is_user=is_user)
        self.mount(msg_widget)
        self.scroll_end(animate=False)

    def add_streaming_message(self, sender: str) -> ChatMessage:
        """Add a streaming message that can be updated incrementally."""
        msg = ChatMessage(sender, "", is_user=False)
        self.mount(msg)
        self.scroll_end(animate=False)
        return msg


class ChatApp(App):
    """Main Textual application for chat interface."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
    }

    #character-card {
        width: 35;
        height: 100%;
        border-right: solid $primary;
        padding: 1;
    }

    #chat-container {
        width: 1fr;
        height: 100%;
    }

    #chat-log {
        height: 1fr;
        border: solid $accent;
        padding: 1;
        margin: 1;
    }

    ChatMessage {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #input-container {
        height: auto;
        padding: 0 1;
    }

    Input {
        width: 100%;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("f2", "toggle_sidebar_mode", "Toggle Sidebar"),
    ]

    def __init__(self) -> None:
        """Initialize the chat application."""
        super().__init__()
        self.conversation_manager: ConversationManager | None = None
        self.character_card_widget: CharacterCard | None = None
        self.chat_log_widget: ChatLog | None = None
        self.status_widget: Static | None = None
        self.input_widget: Input | None = None
        self.is_processing = False

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with Horizontal(id="main-container"):
            self.character_card_widget = CharacterCard()
            yield Container(self.character_card_widget, id="character-card")
            with Vertical(id="chat-container"):
                self.chat_log_widget = ChatLog(id="chat-log")
                yield self.chat_log_widget
                with Container(id="input-container"):
                    self.status_widget = Static("Initializing...", id="status")
                    yield self.status_widget
                    self.input_widget = Input(placeholder="Type your message and press Enter...", id="user-input")
                    self.input_widget.disabled = True
                    yield self.input_widget
        yield Footer()

    async def on_mount(self) -> None:
        """Handle mount event - initialize conversation manager."""
        await self.load_conversation_manager()

    async def load_conversation_manager(self) -> None:
        """Load the conversation manager."""
        self.update_status("Loading model...")
        app_config = load_app_config()
        configure_logging(app_config)

        # Load in thread to avoid blocking the UI
        self.conversation_manager = await asyncio.to_thread(ConversationManager)

        # Update UI with loaded data
        if self.character_card_widget and self.conversation_manager:
            model_name = str(self.conversation_manager.configs.get("MODEL", ""))
            model_type = str(self.conversation_manager.configs.get("MODEL_TYPE", ""))
            rag_collection = str(self.conversation_manager.rag_collection)
            summary = str(self.conversation_manager.description).strip()
            self.character_card_widget.update_character(
                self.conversation_manager.character_name,
                model_name,
                model_type,
                rag_collection,
                summary,
            )

        # Display first message if available
        if self.conversation_manager and self.conversation_manager.first_message and self.chat_log_widget:
            self.chat_log_widget.add_message(
                self.conversation_manager.character_name,
                self.conversation_manager.first_message,
                is_user=False,
            )

        # Enable input
        if self.input_widget:
            self.input_widget.disabled = False
            self.input_widget.focus()

        self.update_status("Ready")

    def update_status(self, message: str) -> None:
        """Update status message."""
        if self.status_widget:
            self.status_widget.update(message)

    @on(Input.Submitted)
    async def handle_input(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if self.is_processing or not self.conversation_manager:
            return

        user_message = event.value.strip()
        if not user_message:
            return

        # Clear input
        if self.input_widget:
            self.input_widget.value = ""

        # Display user message
        if self.chat_log_widget:
            self.chat_log_widget.add_message("User", user_message, is_user=True)

        # Process message
        await self.process_message(user_message)

    async def process_message(self, message: str) -> None:
        """Process user message and get AI response."""
        self.is_processing = True
        self.update_status("Thinking...")

        if self.input_widget:
            self.input_widget.disabled = True

        # Create a streaming message widget for the AI response
        ai_message_widget = None
        if self.chat_log_widget and self.conversation_manager:
            ai_message_widget = ChatMessage(self.conversation_manager.character_name, "", is_user=False)
            self.chat_log_widget.mount(ai_message_widget)
            self.chat_log_widget.scroll_end(animate=False)

        # Use a list to share state between async context and sync callback
        message_chunks: list[str] = []

        def append_chunk_ui(chunk: str) -> None:
            """Apply streamed chunk to UI on Textual's main thread."""
            message_chunks.append(chunk)
            if ai_message_widget:
                ai_message_widget.append_text(chunk)
            if self.chat_log_widget:
                self.chat_log_widget.scroll_end(animate=False)

        def stream_callback(chunk: str) -> None:
            """Callback to handle streaming text chunks."""
            self.call_from_thread(append_chunk_ui, chunk)

        try:
            # Run the async conversation in a thread to avoid blocking the event loop
            await asyncio.to_thread(
                self._ask_question_worker,
                message,
                stream_callback,
            )

            self.update_status("Ready")

        except Exception as e:
            logger.exception("Error processing message")
            self.update_status(f"Error: {e!s}")

        finally:
            self.is_processing = False
            if self.input_widget:
                self.input_widget.disabled = False
                self.input_widget.focus()

    def _ask_question_worker(self, message: str, stream_callback: Callable[[str], None]) -> None:
        """Worker to run ask_question in a thread."""
        # Create a new asyncio event loop for this thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            first_token_event = threading.Event()
            loop.run_until_complete(self.conversation_manager.ask_question(message, first_token_event, stream_callback))
        finally:
            loop.close()

    @on(Resize)
    def handle_resize(self, _event: Resize) -> None:
        """Refresh responsive layout when terminal size changes."""
        if self.chat_log_widget:
            self.chat_log_widget.refresh(layout=True)
            self.chat_log_widget.scroll_end(animate=False)
        if self.character_card_widget:
            self.character_card_widget.refresh(layout=True)

    def action_toggle_sidebar_mode(self) -> None:
        """Toggle sidebar between metadata and summary modes."""
        if not self.character_card_widget:
            return
        self.character_card_widget.toggle_mode()
        mode_name = "summary" if self.character_card_widget.show_summary else "metadata"
        self.update_status(f"Sidebar: {mode_name}")


async def main() -> None:
    """Run the chat TUI application."""
    app = ChatApp()
    await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
