"""QGIS-side view of the validated Schema v2 environment report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RegisteredModel:
    model_id: str
    display_name: str
    version: str
    artifact: str
    artifact_path: str
    sha256: str
    enabled: bool


@dataclass(frozen=True)
class RegisteredFusionProfile:
    profile_id: str
    file_path: str
    enabled: bool
    available: bool
    status: str
    strategy: str
    required_model_ids: tuple[str, ...]
    profile: Mapping[str, Any]


class ModelRegistry:
    def __init__(self, effective: Mapping[str, Any]):
        self.schema_version = effective.get("schema_version")
        if self.schema_version != 2:
            raise ValueError("validated environment report must use Schema v2")

        self.runtime = dict(effective.get("runtime") or {})
        self.scaling = dict(effective.get("scaling") or {})
        self.sam3 = dict(effective.get("sam3") or {})
        self.boundary_fitting = dict(effective.get("boundary_fitting") or {})
        self.classes = dict(effective.get("classes") or {})
        self._models: dict[str, RegisteredModel] = {}
        for raw in effective.get("semantic_models") or []:
            model = RegisteredModel(
                model_id=str(raw["model_id"]),
                display_name=str(raw.get("display_name") or raw["model_id"]),
                version=str(raw.get("version") or ""),
                artifact=str(raw.get("artifact") or ""),
                artifact_path=str(raw.get("artifact_path") or ""),
                sha256=str(raw.get("sha256") or ""),
                enabled=bool(raw.get("enabled", True)),
            )
            if model.model_id in self._models:
                raise ValueError(f"duplicate model_id in report: {model.model_id}")
            self._models[model.model_id] = model

        self._profiles: dict[str, RegisteredFusionProfile] = {}
        for raw in effective.get("fusion_profiles") or []:
            profile = RegisteredFusionProfile(
                profile_id=str(raw["profile_id"]),
                file_path=str(raw.get("file_path") or ""),
                enabled=bool(raw.get("enabled", True)),
                available=bool(raw.get("available", False)),
                status=str(raw.get("status") or "invalid"),
                strategy=str(raw.get("strategy") or ""),
                required_model_ids=tuple(raw.get("required_model_ids") or ()),
                profile=dict(raw.get("profile") or {}),
            )
            if profile.profile_id in self._profiles:
                raise ValueError(f"duplicate profile_id in report: {profile.profile_id}")
            self._profiles[profile.profile_id] = profile

    @property
    def models(self) -> tuple[RegisteredModel, ...]:
        return tuple(self._models.values())

    @property
    def profiles(self) -> tuple[RegisteredFusionProfile, ...]:
        return tuple(self._profiles.values())

    def model(self, model_id: str) -> RegisteredModel:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown model_id: {model_id}") from exc

    def profile(self, profile_id: str) -> RegisteredFusionProfile:
        try:
            return self._profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown profile_id: {profile_id}") from exc

    def resolve_selection(
        self,
        selected_model_ids: list[str] | tuple[str, ...],
        profile_id: str | None,
    ) -> tuple[str, ...]:
        selected = list(dict.fromkeys(selected_model_ids))
        for model_id in selected:
            model = self.model(model_id)
            if not model.enabled:
                raise ValueError(f"model is disabled: {model_id}")

        if profile_id:
            profile = self.profile(profile_id)
            if not profile.enabled or not profile.available or profile.status != "approved":
                raise ValueError(f"fusion profile is not runnable: {profile_id}")
            for model_id in profile.required_model_ids:
                model = self.model(model_id)
                if not model.enabled:
                    raise ValueError(f"fusion requires disabled model: {model_id}")
                if model_id not in selected:
                    selected.append(model_id)
        if not selected:
            raise ValueError("at least one semantic model must be selected")
        return tuple(selected)
