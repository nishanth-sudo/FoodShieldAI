import logging
import sys

from pythonjsonlogger.jsonlogger import JsonFormatter


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Suppress noisy libs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(exc_info)s',
        rename_fields={
            'asctime': 'timestamp',
            'levelname': 'level',
            'name': 'logger_name',
        }
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
