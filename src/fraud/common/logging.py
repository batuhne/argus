"""Structured logging (structlog) setup shared by every entrypoint."""

import logging
import sys
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog once at process start; safe to call again."""
    known_levels = logging.getLevelNamesMapping()
    unknown_level = level.upper() not in known_levels
    log_level = known_levels.get(level.upper(), logging.INFO)

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
    if unknown_level:
        get_logger(__name__).warning("unknown_log_level", requested=level, using="INFO")


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger(name))
