import logging

from rag_pageindex.core.config import settings
from rag_pageindex.core.logging import setup_logging
from rag_pageindex.hello_world import HelloWorld

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entrypoint to run the application"""
    # Setup logging
    setup_logging(log_level=settings.log_level)

    # Start the application
    logger.info("Application started")
    HelloWorld.print_hello_world()


if __name__ == "__main__":
    main()
