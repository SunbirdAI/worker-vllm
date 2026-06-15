"""Configure stdlib logging for the API.

A single-line, structured-ish format with request_id when present.
Avoids pulling in structlog/loguru — stdlib only.
"""

import logging
import sys

from api.errors import get_request_id


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = getattr(record, "request_id", None) or get_request_id()
        record.request_id = rid
        return True


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Wipe existing handlers if reloaders attached extras.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level.upper())
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
        )
    )
    root.addHandler(handler)
