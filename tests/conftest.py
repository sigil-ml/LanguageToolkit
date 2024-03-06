from loguru import logger
import pytest
import sys


@pytest.fixture(scope="session", autouse=True)
def setup_logging(request):
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("./tests/tests.log", rotation="10 MB", level="INFO")
    request.config.logger = logger