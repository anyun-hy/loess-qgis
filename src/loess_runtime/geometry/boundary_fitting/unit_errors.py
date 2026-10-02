"""Errors shared by Unit input and geometry execution."""

__all__ = ["UnitRuntimeError"]


class UnitRuntimeError(RuntimeError):
    """A Unit cannot be read, fitted, or published under its frozen contract."""
