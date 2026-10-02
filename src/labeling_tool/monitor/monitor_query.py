"""Pure request state and transport contracts for the inference monitor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Mapping, NotRequired, TypedDict

QueryKind = Literal["snapshot", "detail", "history", "object_history"]


class QueryIdentityFields(TypedDict):
    generation: int
    request_id: int
    run_id: str


class QueryIdentity(QueryIdentityFields):
    kind: QueryKind


class QueryFailure(QueryIdentity):
    error: str


class _RequestFields(QueryIdentityFields):
    run_spec: dict[str, object]


class SnapshotRequest(_RequestFields):
    kind: Literal["snapshot"]


class DetailRequest(_RequestFields):
    kind: Literal["detail"]
    stream_id: str
    detail_kind: str
    status: str
    search: str
    page: int
    page_size: int


class HistoryRequest(_RequestFields):
    kind: Literal["history"]
    scope: str
    execution_id: str
    search: str
    before_event_id: int | None
    append: bool
    page_size: int
    context: dict[str, object]
    stream_id: str
    object_id: str
    job_id: object
    span_id: str


class ObjectHistoryRequest(_RequestFields):
    kind: Literal["object_history"]
    object_id: str
    detail_kind: str
    stream_id: str
    job_id: object
    span_id: str
    append: bool
    before_started_at: str
    before_span_id: str


QueryMessage = SnapshotRequest | DetailRequest | HistoryRequest | ObjectHistoryRequest


class SnapshotResult(QueryIdentityFields):
    kind: Literal["snapshot"]
    snapshot: dict[str, object]


class DetailResult(QueryIdentityFields):
    kind: Literal["detail"]
    stream_id: str
    detail_kind: str
    status: str
    search: str
    page: int
    page_size: int
    page_total: int
    total: int
    rows: list[dict[str, object]]


class HistoryResult(QueryIdentityFields):
    kind: Literal["history"]
    append: bool
    rows: list[dict[str, object]]
    page_size: int


class RawLogPage(TypedDict):
    raw_log: Literal[True]
    rows: list[dict[str, object]]
    page_size: NotRequired[int]
    next_cursor: int | None
    has_more: bool
    skipped_records: int


class RawHistoryResult(QueryIdentityFields, RawLogPage):
    kind: Literal["history"]
    append: bool


class ObjectHistoryResult(QueryIdentityFields):
    kind: Literal["object_history"]
    object_id: str
    detail_kind: str
    spans: list[dict[str, object]]
    events: list[dict[str, object]]
    models: list[dict[str, object]]
    attempt_id: str
    append: bool
    has_more: bool


QueryResult = (
    SnapshotResult
    | DetailResult
    | HistoryResult
    | RawHistoryResult
    | ObjectHistoryResult
)


@dataclass(frozen=True)
class QueryCompletion:
    """The active request released by an exact worker callback."""

    request: QueryMessage
    applicable: bool


_PRIORITY: tuple[QueryKind, ...] = (
    "detail",
    "object_history",
    "history",
    "snapshot",
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _plain(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


def query_filter(request: Mapping[str, object]) -> tuple[object, ...]:
    """Return the UI-relevant filter fingerprint for one request."""

    kind = str(request.get("kind") or "")
    if kind == "detail":
        return tuple(
            request.get(name)
            for name in (
                "stream_id",
                "detail_kind",
                "status",
                "search",
                "page",
                "page_size",
            )
        )
    if kind == "history":
        scope = str(request.get("scope") or "all")
        if scope.startswith("raw_"):
            return (scope, request.get("search") or "")
        return (
            scope,
            request.get("execution_id") or "",
            request.get("search") or "",
            _plain(request.get("context") or {}),
        )
    if kind == "object_history":
        return tuple(
            request.get(name)
            for name in (
                "object_id",
                "detail_kind",
                "stream_id",
                "job_id",
                "span_id",
            )
        )
    return ()


def query_matches_filter(
    request: Mapping[str, object],
    current_filter: Mapping[str, object],
) -> bool:
    """Compare a request against one current UI filter without mutable state."""

    current = dict(current_filter)
    current["kind"] = request.get("kind")
    return query_filter(request) == query_filter(current)


class MonitorQueryCoordinator:
    """One-in-flight, latest-wins request coordinator with no I/O or Qt."""

    def __init__(self) -> None:
        self._run_id = ""
        self._run_spec: dict[str, object] = {}
        self._generation = 0
        self._request_serial = 0
        self._latest_request_ids: dict[QueryKind, int] = {kind: 0 for kind in _PRIORITY}
        self._pending: dict[QueryKind, QueryMessage | None] = {
            kind: None for kind in _PRIORITY
        }
        self._active: QueryMessage | None = None

    @property
    def is_bound(self) -> bool:
        return bool(self._run_id)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def generation(self) -> int:
        return self._generation

    def run_spec(self) -> dict[str, object]:
        return deepcopy(self._run_spec)

    def bind(self, run_id: str, run_spec: Mapping[str, object]) -> None:
        self._advance_generation()
        self._run_id = str(run_id)
        self._run_spec = deepcopy(dict(run_spec))

    def unbind(self) -> None:
        self._advance_generation()
        self._run_id = ""
        self._run_spec = {}

    def fence(self) -> None:
        """Discard queued work while preserving an active worker call."""

        self._advance_generation()

    def _advance_generation(self) -> None:
        self._generation += 1
        for kind in _PRIORITY:
            self._latest_request_ids[kind] = 0
            self._pending[kind] = None

    def _request_fields(self) -> _RequestFields:
        return {
            "generation": self._generation,
            "request_id": self._request_serial + 1,
            "run_id": self._run_id,
            "run_spec": self.run_spec(),
        }

    def _enqueue(self, request: QueryMessage) -> int | None:
        if not self.is_bound:
            return None
        request_id = request["request_id"]
        self._request_serial = request_id
        kind = request["kind"]
        self._latest_request_ids[kind] = request_id
        self._pending[kind] = request
        return request_id

    def enqueue_snapshot(self) -> int | None:
        request: SnapshotRequest = {**self._request_fields(), "kind": "snapshot"}
        return self._enqueue(request)

    def enqueue_detail(
        self,
        *,
        stream_id: str,
        detail_kind: str,
        status: str,
        search: str,
        page: int,
        page_size: int,
    ) -> int | None:
        request: DetailRequest = {
            **self._request_fields(),
            "kind": "detail",
            "stream_id": str(stream_id),
            "detail_kind": str(detail_kind),
            "status": str(status),
            "search": str(search),
            "page": int(page),
            "page_size": int(page_size),
        }
        return self._enqueue(request)

    def enqueue_history(
        self,
        *,
        scope: str,
        execution_id: str,
        search: str,
        before_event_id: int | None,
        append: bool,
        page_size: int,
        context: Mapping[str, object],
    ) -> int | None:
        request: HistoryRequest = {
            **self._request_fields(),
            "kind": "history",
            "scope": str(scope),
            "execution_id": str(execution_id),
            "search": str(search),
            "before_event_id": before_event_id,
            "append": bool(append),
            "page_size": int(page_size),
            "context": deepcopy(dict(context)),
            "stream_id": str(context.get("stream_id") or ""),
            "object_id": str(context.get("object_id") or ""),
            "job_id": context.get("job_id"),
            "span_id": str(context.get("span_id") or ""),
        }
        return self._enqueue(request)

    def enqueue_object_history(
        self,
        *,
        object_id: str,
        detail_kind: str,
        stream_id: str,
        job_id: object,
        span_id: str,
        append: bool,
        before_started_at: str,
        before_span_id: str,
    ) -> int | None:
        request: ObjectHistoryRequest = {
            **self._request_fields(),
            "kind": "object_history",
            "object_id": str(object_id),
            "detail_kind": str(detail_kind),
            "stream_id": str(stream_id),
            "job_id": job_id,
            "span_id": str(span_id),
            "append": bool(append),
            "before_started_at": str(before_started_at),
            "before_span_id": str(before_span_id),
        }
        return self._enqueue(request)

    def take_next(self) -> QueryMessage | None:
        if self._active is not None:
            return None
        for kind in _PRIORITY:
            request = self._pending[kind]
            if request is not None:
                self._pending[kind] = None
                self._active = request
                return deepcopy(request)
        return None

    def complete(self, message: Mapping[str, object]) -> QueryCompletion | None:
        active = self._active
        if active is None or not self._same_identity(message, active):
            return None
        self._active = None
        applicable = (
            self.is_bound
            and active["generation"] == self._generation
            and str(active["run_id"]) == self._run_id
        )
        return QueryCompletion(request=deepcopy(active), applicable=applicable)

    def continuation_is_current(
        self,
        kind: QueryKind,
        *,
        generation: int,
        request_id: int,
    ) -> bool:
        return (
            self.is_bound
            and generation == self._generation
            and request_id == self._latest_request_ids[kind]
        )

    @staticmethod
    def _same_identity(
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> bool:
        return all(
            left.get(name) == right.get(name)
            for name in ("kind", "generation", "request_id", "run_id")
        )
