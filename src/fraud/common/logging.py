import logging
import sys
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog once at process start.

    Console rendering for local work, JSON for containers where logs are
    scraped. Safe to call more than once.
    """
    log_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger(name))
