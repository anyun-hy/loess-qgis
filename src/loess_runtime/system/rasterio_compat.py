"""Narrow compatibility handling for Rasterio with GDAL 3.11+."""

from __future__ import annotations

from contextlib import contextmanager
import logging

import rasterio


_MEMORY_DRIVER_WARNING = "'Memory' driver is deprecated since GDAL 3.11"


class _MemoryDriverWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _MEMORY_DRIVER_WARNING not in record.getMessage()


@contextmanager
def quiet_deprecated_memory_driver():
    """Hide Rasterio's known Memory-driver warning and preserve all other errors."""

    logger = logging.getLogger("rasterio._env")
    warning_filter = _MemoryDriverWarningFilter()
    logger.addFilter(warning_filter)
    try:
        with rasterio.Env(CPL_LOG_ERRORS=False):
            yield
    finally:
        logger.removeFilter(warning_filter)
