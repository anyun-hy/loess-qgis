"""
Unified device resolution for inference scripts.
"""
try:
    import torch
except ImportError:
    torch = None

_HAS_TORCH = torch is not None


def mps_available():
    return (
        _HAS_TORCH
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def resolve_device(configured_device: str) -> str:
    """Resolve a device string. 'auto' → cuda/mps/cpu based on availability."""
    requested = str(configured_device or "auto").strip().lower()
    if requested == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda:0"
        if mps_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        return "cuda:0"
    if requested == "mps:0":
        return "mps"
    return requested


def validate_device(device: str) -> bool:
    """Check if a device string is available on the system."""
    normalized = str(device or "").strip().lower()
    if normalized.startswith("cuda"):
        if not _HAS_TORCH or not torch.cuda.is_available():
            return False
        if normalized == "cuda":
            device_index = 0
        else:
            try:
                prefix, raw_index = normalized.split(":", 1)
                if prefix != "cuda":
                    return False
                device_index = int(raw_index)
            except (TypeError, ValueError):
                return False
        return 0 <= device_index < torch.cuda.device_count()
    if normalized == "mps":
        return mps_available()
    return normalized == "cpu"
