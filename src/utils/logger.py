from __future__ import annotations

import atexit
import json
import logging
from typing import Any


class Logger:
    _logger: logging.Logger | None = None
    _initialized = False

    @classmethod
    def init(cls, channel_name: str = "mnemo") -> None:
        if cls._initialized:
            return

        logger = logging.getLogger(channel_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        cls._logger = logger

        cls._initialized = True
        atexit.register(cls.dispose)

    @classmethod
    def dispose(cls) -> None:
        if cls._logger is not None:
            handlers = list(cls._logger.handlers)
            for handler in handlers:
                handler.flush()
                handler.close()
                cls._logger.removeHandler(handler)
            cls._logger = None
        cls._initialized = False

    @classmethod
    def debug(cls, message: str, source: str | None = None, data: dict[str, Any] | None = None) -> None:
        cls._log("debug", message, source, data)

    @classmethod
    def info(cls, message: str, source: str | None = None, data: dict[str, Any] | None = None) -> None:
        cls._log("info", message, source, data)

    @classmethod
    def warn(cls, message: str, source: str | None = None, data: dict[str, Any] | None = None) -> None:
        cls._log("warning", message, source, data)

    @classmethod
    def error(cls, message: str, source: str | None = None, data: dict[str, Any] | None = None) -> None:
        cls._log("error", message, source, data)

    @classmethod
    def _log(
        cls,
        level: str,
        message: str,
        source: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not cls._initialized:
            cls.init()

        src = f"[{source}] " if source else ""
        metadata = f"\n{json.dumps(data, indent=2, ensure_ascii=False)}" if data is not None else ""
        payload = f"{src}{message}{metadata}"

        assert cls._logger is not None
        getattr(cls._logger, level)(payload)
