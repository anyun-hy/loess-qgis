"""Adaptive memory admission for the QGIS-owned inference process group.

Linux ``systemd-oomd`` reacts to sustained reclaim stalls (PSI), not merely to
an RSS byte limit.  The scheduler therefore needs both an early pressure signal
and a gradual concurrency ramp.  This module stays independent of Qt so the
policy and the cgroup-v2 sampler can be tested without a QGIS process.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Callable, Mapping


GIB = 1024**3


DEFAULT_MEMORY_ADMISSION_POLICY = {
    "schema_version": 1,
    "mode": "adaptive_psi_aimd_v1",
    "sample_interval_sec": 1.0,
    "stable_growth_sec": 5.0,
    "pressure_cooldown_sec": 5.0,
    "initial_geometry_slots_with_package": 4,
    "initial_geometry_slots_without_package": 8,
    "default_worker_peak_bytes": int(2.5 * GIB),
    "minimum_available_reserve_bytes": 16 * GIB,
    "available_reserve_ratio": 0.20,
    "pressure_some_avg10": 8.0,
    "pressure_full_avg10": 1.0,
    "severe_some_avg10": 25.0,
    "severe_full_avg10": 5.0,
    "stable_some_avg10": 2.0,
    "stable_full_avg10": 0.2,
    "low_available_ratio": 0.20,
    "severe_available_ratio": 0.10,
    "high_swap_used_ratio": 0.65,
    "severe_swap_used_ratio": 0.85,
}


@dataclass(frozen=True)
class MemoryPressureSample:
    """One cheap host/cgroup memory observation."""

    supported: bool = False
    total_bytes: int = 0
    available_bytes: int = 0
    swap_total_bytes: int = 0
    swap_free_bytes: int = 0
    cgroup_current_bytes: int = 0
    cgroup_anon_bytes: int = 0
    cgroup_file_bytes: int = 0
    some_avg10: float = 0.0
    full_avg10: float = 0.0
    pressure_source_count: int = 0

    @property
    def available_ratio(self) -> float:
        if self.total_bytes <= 0:
            return 1.0
        return max(0.0, min(1.0, self.available_bytes / self.total_bytes))

    @property
    def swap_used_ratio(self) -> float:
        if self.swap_total_bytes <= 0:
            return 0.0
        used = max(0, self.swap_total_bytes - self.swap_free_bytes)
        return max(0.0, min(1.0, used / self.swap_total_bytes))

    def payload(self) -> dict[str, object]:
        return {
            "supported": bool(self.supported),
            "total_bytes": int(self.total_bytes),
            "available_bytes": int(self.available_bytes),
            "available_ratio": round(self.available_ratio, 4),
            "swap_total_bytes": int(self.swap_total_bytes),
            "swap_free_bytes": int(self.swap_free_bytes),
            "swap_used_ratio": round(self.swap_used_ratio, 4),
            "cgroup_current_bytes": int(self.cgroup_current_bytes),
            "cgroup_anon_bytes": int(self.cgroup_anon_bytes),
            "cgroup_file_bytes": int(self.cgroup_file_bytes),
            "some_avg10": round(float(self.some_avg10), 3),
            "full_avg10": round(float(self.full_avg10), 3),
            "pressure_source_count": int(self.pressure_source_count),
        }


@dataclass(frozen=True)
class MemoryAdmissionDecision:
    """Dynamic geometry budget returned to the scheduler."""

    geometry_slot_limit: int
    pause_new_work: bool
    shed_active_work: bool
    reason: str
    worker_peak_estimate_bytes: int
    changed: bool
    sample: MemoryPressureSample

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "geometry_slot_limit": int(self.geometry_slot_limit),
            "pause_new_work": bool(self.pause_new_work),
            "shed_active_work": bool(self.shed_active_work),
            "reason": str(self.reason),
            "worker_peak_estimate_bytes": int(self.worker_peak_estimate_bytes),
            "sample": self.sample.payload(),
        }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        values[name] = value * multiplier
    return values


def _parse_pressure(text: str) -> tuple[float, float]:
    result = {"some": 0.0, "full": 0.0}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in result:
            continue
        for field in fields[1:]:
            name, separator, value = field.partition("=")
            if name != "avg10" or not separator:
                continue
            try:
                result[fields[0]] = max(0.0, float(value))
            except ValueError:
                pass
            break
    return result["some"], result["full"]


def _parse_memory_stat(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            values[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return values


def _unified_cgroup_path(text: str) -> str:
    for line in text.splitlines():
        hierarchy, controllers, path = (line.split(":", 2) + ["", ""])[:3]
        if hierarchy == "0" and controllers == "":
            return path
    return ""


def read_linux_memory_pressure(
    *,
    proc_root: str | Path = "/proc",
    cgroup_root: str | Path = "/sys/fs/cgroup",
) -> MemoryPressureSample:
    """Read host memory and the strongest PSI value in our cgroup ancestry."""

    proc = Path(proc_root)
    cgroup = Path(cgroup_root)
    meminfo = _parse_meminfo(_read_text(proc / "meminfo"))
    total = int(meminfo.get("MemTotal", 0))
    available = int(meminfo.get("MemAvailable", 0))
    if total <= 0 or "MemAvailable" not in meminfo:
        return MemoryPressureSample()

    pressure_values: list[tuple[float, float]] = []
    system_pressure = _read_text(proc / "pressure" / "memory")
    if system_pressure:
        pressure_values.append(_parse_pressure(system_pressure))

    relative = _unified_cgroup_path(_read_text(proc / "self" / "cgroup"))
    leaf = cgroup / relative.lstrip("/") if relative else None
    current = anon = file_bytes = 0
    if leaf is not None and leaf.exists():
        try:
            current = int(_read_text(leaf / "memory.current").strip() or 0)
        except ValueError:
            current = 0
        memory_stat = _parse_memory_stat(_read_text(leaf / "memory.stat"))
        anon = int(memory_stat.get("anon", 0))
        file_bytes = int(memory_stat.get("file", 0))
        candidate = leaf
        while candidate == cgroup or cgroup in candidate.parents:
            pressure_text = _read_text(candidate / "memory.pressure")
            if pressure_text:
                pressure_values.append(_parse_pressure(pressure_text))
            if candidate == cgroup:
                break
            candidate = candidate.parent

    some = max((value[0] for value in pressure_values), default=0.0)
    full = max((value[1] for value in pressure_values), default=0.0)
    return MemoryPressureSample(
        supported=True,
        total_bytes=total,
        available_bytes=available,
        swap_total_bytes=int(meminfo.get("SwapTotal", 0)),
        swap_free_bytes=int(meminfo.get("SwapFree", 0)),
        cgroup_current_bytes=current,
        cgroup_anon_bytes=anon,
        cgroup_file_bytes=file_bytes,
        some_avg10=some,
        full_avg10=full,
        pressure_source_count=len(pressure_values),
    )


class AdaptiveMemoryAdmissionController:
    """AIMD controller with PSI-triggered pause and non-blocking load shedding."""

    def __init__(
        self,
        policy: Mapping[str, object] | None = None,
        *,
        sampler: Callable[[], MemoryPressureSample] = read_linux_memory_pressure,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._policy = dict(DEFAULT_MEMORY_ADMISSION_POLICY)
        if isinstance(policy, Mapping):
            self._policy.update(policy)
        self._sampler = sampler
        self._clock = clock
        self._limit: int | None = None
        self._last_sample = MemoryPressureSample()
        self._last_sample_at = float("-inf")
        self._last_growth_at = 0.0
        self._last_reduction_at = float("-inf")
        self._last_reduction_sample_at = float("-inf")
        self._last_reduction_severe = False
        self._worker_peaks: list[int] = []
        self._completed_observation_count = 0
        self._completed_observation_count_at_growth = 0
        self._last_signature: tuple[int, bool, bool] | None = None

    def observe_worker_peak(self, peak_bytes: int) -> None:
        """Add one successful unit-fit peak without trusting malformed events."""

        try:
            value = int(peak_bytes)
        except (TypeError, ValueError):
            return
        if not 32 * 1024**2 <= value <= 64 * GIB:
            return
        self._worker_peaks.append(value)
        del self._worker_peaks[:-64]
        self._completed_observation_count += 1

    def _worker_peak_estimate(self) -> int:
        default = max(1, int(self._policy["default_worker_peak_bytes"]))
        if not self._worker_peaks:
            return default
        ordered = sorted(self._worker_peaks)
        index = max(0, math.ceil(len(ordered) * 0.90) - 1)
        observed = int(ordered[index] * 1.20)
        return max(default, observed)

    def _sample(self, now: float) -> MemoryPressureSample:
        interval = max(0.1, float(self._policy["sample_interval_sec"]))
        if now - self._last_sample_at < interval:
            return self._last_sample
        try:
            sample = self._sampler()
        except Exception:
            sample = MemoryPressureSample()
        if not isinstance(sample, MemoryPressureSample):
            sample = MemoryPressureSample()
        self._last_sample = sample
        self._last_sample_at = now
        return sample

    def decide(
        self,
        *,
        static_limit: int,
        active_slots: int,
        package_active: bool,
        now: float | None = None,
        sample: MemoryPressureSample | None = None,
    ) -> MemoryAdmissionDecision:
        """Return the current safe geometry budget without blocking the caller."""

        observed_at = self._clock() if now is None else float(now)
        ceiling = max(1, int(static_limit))
        active = max(0, int(active_slots))
        current_sample = sample if sample is not None else self._sample(observed_at)
        sample_at = observed_at if sample is not None else self._last_sample_at
        fresh_reduction_sample = sample_at > self._last_reduction_sample_at
        worker_estimate = self._worker_peak_estimate()

        if not current_sample.supported:
            limit = ceiling
            pause = shed = False
            reason = "pressure_sensor_unavailable_static_fallback"
            self._limit = limit
        else:
            if self._limit is None:
                initial_name = (
                    "initial_geometry_slots_with_package"
                    if package_active
                    else "initial_geometry_slots_without_package"
                )
                self._limit = min(ceiling, max(1, int(self._policy[initial_name])))
                self._last_growth_at = observed_at
            self._limit = min(ceiling, max(1, int(self._limit)))

            available_ratio = current_sample.available_ratio
            swap_ratio = current_sample.swap_used_ratio
            some = float(current_sample.some_avg10)
            full = float(current_sample.full_avg10)
            severe = (
                some >= float(self._policy["severe_some_avg10"])
                or full >= float(self._policy["severe_full_avg10"])
                or available_ratio <= float(self._policy["severe_available_ratio"])
                or swap_ratio >= float(self._policy["severe_swap_used_ratio"])
            )
            pressured = severe or (
                some >= float(self._policy["pressure_some_avg10"])
                or full >= float(self._policy["pressure_full_avg10"])
                or available_ratio <= float(self._policy["low_available_ratio"])
                or swap_ratio >= float(self._policy["high_swap_used_ratio"])
            )
            stable = (
                not pressured
                and some < float(self._policy["stable_some_avg10"])
                and full < float(self._policy["stable_full_avg10"])
                and available_ratio > float(self._policy["low_available_ratio"])
            )

            reserve = max(
                int(self._policy["minimum_available_reserve_bytes"]),
                int(
                    current_sample.total_bytes
                    * float(self._policy["available_reserve_ratio"])
                ),
            )
            headroom = max(0, current_sample.available_bytes - reserve)
            memory_ceiling = max(active, active + headroom // worker_estimate)

            pause = shed = False
            reduction_cooled = observed_at - self._last_reduction_at >= float(
                self._policy["pressure_cooldown_sec"]
            )
            if severe:
                if fresh_reduction_sample and (not self._last_reduction_severe or reduction_cooled):
                    self._limit = max(1, min(self._limit, math.ceil(max(active, 1) / 4)))
                    self._last_reduction_at = observed_at
                    self._last_reduction_sample_at = sample_at
                    self._last_reduction_severe = True
                pause = True
                shed = active > self._limit
                reason = "severe_memory_pressure"
                self._last_growth_at = observed_at
            elif pressured:
                if fresh_reduction_sample and reduction_cooled:
                    self._limit = max(1, min(self._limit, math.ceil(max(active, 1) / 2)))
                    self._last_reduction_at = observed_at
                    self._last_reduction_sample_at = sample_at
                    self._last_reduction_severe = False
                pause = True
                # Backpressure first: transient PSI must not kill writers that
                # are draining the queue. Only severe pressure permits shedding.
                shed = False
                reason = "memory_pressure"
                self._last_growth_at = observed_at
            elif (
                stable
                and active >= self._limit
                and self._completed_observation_count
                > self._completed_observation_count_at_growth
                and observed_at - self._last_growth_at
                >= float(self._policy["stable_growth_sec"])
            ):
                self._limit = min(ceiling, self._limit + 1)
                self._last_growth_at = observed_at
                self._completed_observation_count_at_growth = (
                    self._completed_observation_count
                )
                reason = "stable_growth"
            elif stable:
                reason = "stable_warmup"
            else:
                reason = "holding"

            limit = max(1, min(ceiling, self._limit, int(memory_ceiling)))
            if limit < self._limit:
                self._limit = limit
                if active > limit:
                    pause = True
                    shed = True
                reason = "available_memory_budget"

        signature = (int(limit), bool(pause), bool(shed))
        changed = signature != self._last_signature
        self._last_signature = signature
        return MemoryAdmissionDecision(
            geometry_slot_limit=int(limit),
            pause_new_work=bool(pause),
            shed_active_work=bool(shed),
            reason=str(reason),
            worker_peak_estimate_bytes=worker_estimate,
            changed=changed,
            sample=current_sample,
        )
