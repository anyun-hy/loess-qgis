"""Shared failures for stream assembly responsibilities."""

from __future__ import annotations


class StreamAssemblyError(RuntimeError):
    """A validated stream-assembly operation could not complete."""
