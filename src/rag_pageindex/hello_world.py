import logging

from rag_pageindex.core.config import settings

logger = logging.getLogger(__name__)


class HelloWorld:
    """A simple class to print a hello world message"""

    @staticmethod
    def print_hello_world() -> None:
        """Print a hello world message"""
        logger.info("Hello world function started")
        print(
            "Hello world (project : Rag Pageindex,"
            f" env : {settings.environment})"
        )
        logger.info("Hello world function ended")
