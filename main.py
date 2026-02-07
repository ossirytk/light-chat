import asyncio
import logging

from conversation_manager import ConversationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conversation_manager = ConversationManager()


async def main() -> None:
    """Main interactive chat loop."""
    while True:
        query = input("User: ")
        await conversation_manager.ask_question_test(query)
        print()  # noqa: T201


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Caught keyboard interrupt. Exiting...")


