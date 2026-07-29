"""Crash-safe, partition-local incremental fusion accumulators."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class IncrementalFusionError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=0, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float32)
    return exponent / exponent.sum(axis=0, keepdims=True)


class FusionAccumulator:
    def __init__(self, root: str | Path, profile: Mapping[str, Any], shape: tuple[int, int, int]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile = dict(profile)
        self.model_entries = list(profile.get("models") or [])
        self.model_ids = [str(item["model_id"]) for item in self.model_entries]
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 3 or self.shape[0] != 14 or min(self.shape) < 1:
            raise IncrementalFusionError("fusion shape must be [14,H,W]")
        if not self.model_ids or len(set(self.model_ids)) != len(self.model_ids):
            raise IncrementalFusionError("fusion profile model IDs are empty or duplicated")
        self.strategy = str(profile.get("strategy") or "")
        if self.strategy not in {
            "equal_probability_average",
            "calibrated_global_weighted",
            "calibrated_class_weighted",
            "linear_1x1",
        }:
            raise IncrementalFusionError(f"unsupported fusion strategy: {self.strategy}")
        self.state_path = self.root / "state.json"
        if not self.state_path.exists():
            _atomic_json(
                self.state_path,
                {
                    "schema_version": 1,
                    "strategy": self.strategy,
                    "model_ids": self.model_ids,
                    "shape": list(self.shape),
                    "completed_model_ids": [],
                    "active_array": "",
                    "finalized": False,
                },
            )
        self._validate_state(self._state())

    def _state(self) -> dict[str, Any]:
        with open(self.state_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("strategy") != self.strategy
            or state.get("model_ids") != self.model_ids
            or state.get("shape") != list(self.shape)
        ):
            raise IncrementalFusionError("fusion accumulator state does not match profile")

    def add_model(self, model_id: str, probabilities: np.ndarray) -> dict[str, Any]:
        state = self._state()
        self._validate_state(state)
        completed = list(state["completed_model_ids"])
        if state.get("finalized"):
            raise IncrementalFusionError("fusion accumulator is already finalized")
        if model_id in completed:
            return state
        expected = self.model_ids[len(completed)] if len(completed) < len(self.model_ids) else None
        if model_id != expected:
            raise IncrementalFusionError(f"expected next fusion model {expected}, got {model_id}")
        values = np.asarray(probabilities, dtype=np.float32)
        if values.shape != self.shape or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise IncrementalFusionError("model probabilities violate shape or value contract")
        sums = values.sum(axis=0)
        covered = sums > 0
        if np.any(covered) and not np.allclose(
            sums[covered], 1.0, atol=5e-3, rtol=0
        ):
            raise IncrementalFusionError(
                "covered model probabilities do not sum to one"
            )

        previous_path = self.root / str(state.get("active_array") or "")
        if completed and not previous_path.is_file():
            raise IncrementalFusionError("active fusion accumulator array is missing")
        if self.strategy == "linear_1x1":
            output_shape = (len(self.model_ids) * 14, self.shape[1], self.shape[2])
            next_array = np.zeros(output_shape, dtype=np.float32)
            if completed:
                next_array[:] = np.load(previous_path, allow_pickle=False)
            entry = self.model_entries[len(completed)]
            temperature = float(entry["temperature"])
            weights = np.asarray(self.profile["weights"], dtype=np.float32)[:, len(completed)]
            calibrated = np.log(np.clip(values, 1e-7, 1.0)) / temperature
            start = len(completed) * 14
            next_array[start : start + 14] = calibrated * weights[:, None, None]
        else:
            next_array = (
                np.load(previous_path, allow_pickle=False).astype(np.float32)
                if completed
                else np.zeros(self.shape, dtype=np.float32)
            )
            if self.strategy == "equal_probability_average":
                next_array += values
            else:
                entry = self.model_entries[len(completed)]
                temperature = float(entry["temperature"])
                weights = np.asarray(self.profile["weights"], dtype=np.float32)[:, len(completed)]
                next_array += (
                    np.log(np.clip(values, 1e-7, 1.0))
                    / temperature
                    * weights[:, None, None]
                )
        generation = len(completed) + 1
        next_name = f"accumulator_{generation:03d}.npy"
        temporary = self.root / f".{next_name}.tmp"
        with open(temporary, "wb") as handle:
            np.save(handle, next_array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.root / next_name)
        state = {
            **state,
            "completed_model_ids": completed + [model_id],
            "active_array": next_name,
        }
        _atomic_json(self.state_path, state)
        if completed and previous_path.is_file():
            previous_path.unlink()
        return state

    def finalize(self, *, fusion_head=None) -> np.ndarray:
        state = self._state()
        self._validate_state(state)
        if state["completed_model_ids"] != self.model_ids:
            raise IncrementalFusionError("cannot finalize before every profile model is committed")
        values = np.load(self.root / state["active_array"], allow_pickle=False).astype(np.float32)
        if self.strategy == "equal_probability_average":
            probabilities = values / float(len(self.model_ids))
        elif self.strategy in {"calibrated_global_weighted", "calibrated_class_weighted"}:
            probabilities = _softmax(values)
        else:
            if fusion_head is None:
                raise IncrementalFusionError("linear_1x1 finalization requires fusion_head")
            output = fusion_head(values[None, ...])
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()
            output = np.asarray(output, dtype=np.float32)
            if output.shape != (1, 14, self.shape[1], self.shape[2]):
                raise IncrementalFusionError("fusion_head output must be [1,14,H,W]")
            probabilities = _softmax(output[0])
        state = {**state, "finalized": True}
        _atomic_json(self.state_path, state)
        return probabilities.astype(np.float32, copy=False)
