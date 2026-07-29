"""Core package with lazy QGIS runtime imports.

Pure configuration and profile modules must remain importable in the Conda test
environment where the QGIS Python package is not installed.
"""

__all__ = ("V5AsyncInferenceRunner",)


def __getattr__(name):
    if name == "V5AsyncInferenceRunner":
        from .v5_async_runner import V5AsyncInferenceRunner

        return V5AsyncInferenceRunner
    raise AttributeError(name)
