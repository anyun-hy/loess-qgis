"""Connection and transaction ownership for PostgreSQL Run state."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from labeling_tool.shared.contracts.monitor_contract import MONITOR_EXECUTION_ENV
from labeling_tool.shared.state.postgres_state import (
    DEFAULT_POSTGRES_DSN,
    DEFAULT_POSTGRES_SCHEMA,
    PostgresConnection,
    connect_postgres,
    is_postgres_location,
    validate_schema,
)
from labeling_tool.shared.state.run_execution_ownership import (
    RunExecutionIdentity,
    RunOwnershipLostError,
    execution_identity_from_environment,
)

STATE_DB_DSN_ENV = "LOESS_STATE_DB_DSN"
STATE_DB_SCHEMA_ENV = "LOESS_STATE_DB_SCHEMA"


class RunStateError(RuntimeError):
    """A frozen Run-state or transaction contract was rejected."""


class _ExecutionIdentity:
    def __init__(self, value: str) -> None:
        self.value = str(value).strip()


def production_state_database() -> str:
    """Return the password-free PostgreSQL DSN frozen into new Run Specs."""
    return str(os.environ.get(STATE_DB_DSN_ENV) or DEFAULT_POSTGRES_DSN).strip()


def production_state_schema() -> str:
    """Return the validated PostgreSQL schema for new Run Specs."""
    return str(
        validate_schema(os.environ.get(STATE_DB_SCHEMA_ENV) or DEFAULT_POSTGRES_SCHEMA)
    )


class RunStateSession:
    """Own connection policy and optional scoped transaction borrowing."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str | None = None,
        _connection: PostgresConnection | None = None,
        _unit_identity: tuple[str, str, str] | None = None,
        _execution: _ExecutionIdentity | None = None,
        _execution_owner: RunExecutionIdentity | None = None,
    ) -> None:
        location = str(dsn).strip()
        if not is_postgres_location(location):
            raise RunStateError(
                "Run state requires a PostgreSQL DSN; filesystem databases are "
                "no longer supported"
            )
        self._location = location
        self._schema = str(validate_schema(schema or production_state_schema()))
        self._bound_connection = _connection
        self._unit_identity = _unit_identity
        self._execution = _execution or _ExecutionIdentity(
            str(os.environ.get(MONITOR_EXECUTION_ENV) or "")
        )
        self._execution_owner = _execution_owner
        if self._execution_owner is None and _connection is None:
            self._execution_owner = execution_identity_from_environment(
                schema=self._schema,
                execution_id=self._execution.value,
            )
        self._active = True

    @property
    def location(self) -> str:
        self._require_active()
        return self._location

    @property
    def schema(self) -> str:
        self._require_active()
        return self._schema

    @property
    def execution_id(self) -> str:
        self._require_active()
        return self._execution.value

    @execution_id.setter
    def execution_id(self, value: str) -> None:
        self._require_active()
        if self._bound_connection is not None:
            raise RunStateError(
                "scoped Run-state sessions cannot replace monitor execution identity"
            )
        self._execution.value = str(value).strip()

    @property
    def unit_identity(self) -> tuple[str, str, str] | None:
        self._require_active()
        return self._unit_identity

    @property
    def execution_owner(self) -> RunExecutionIdentity | None:
        self._require_active()
        return self._execution_owner

    @property
    def is_scoped_transaction(self) -> bool:
        self._require_active()
        return self._bound_connection is not None

    def bind_execution_owner(self, identity: RunExecutionIdentity) -> None:
        self._require_active()
        if self._bound_connection is not None:
            raise RunStateError(
                "scoped Run-state sessions cannot bind execution owners"
            )
        if str(identity.schema) != self._schema:
            raise RunStateError(
                "Run execution owner schema does not match state session"
            )
        if self._execution.value and self._execution.value != identity.execution_id:
            raise RunStateError(
                "Run execution owner does not match monitor execution identity"
            )
        self._execution.value = identity.execution_id
        self._execution_owner = identity

    def connect(self, *, autocommit: bool = True) -> PostgresConnection:
        """Open one owned PostgreSQL connection outside a scoped transaction."""
        self._require_active()
        if self._bound_connection is not None:
            raise RunStateError("scoped Run-state sessions cannot open connections")
        return connect_postgres(
            self._location,
            schema=self._schema,
            autocommit=autocommit,
        )

    @contextlib.contextmanager
    def connection(self) -> Iterator[PostgresConnection]:
        """Yield a borrowed scoped connection or one owned short-lived connection."""
        self._require_active()
        if self._bound_connection is not None:
            yield self._bound_connection
            return
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[PostgresConnection]:
        """Yield one transaction, reusing a scoped unit-publication connection."""
        self._require_active()
        if self._bound_connection is not None:
            yield self._bound_connection
            return
        connection = self.connect(autocommit=False)
        try:
            if self._execution_owner is not None:
                self._execution_owner.assert_current(connection)
            yield connection
            connection.commit()
        except RunOwnershipLostError:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def scoped(
        self,
        connection: PostgresConnection,
        unit_identity: tuple[str, str, str],
    ) -> RunStateSession:
        """Bind a child session to a transaction owned by this active session."""
        self._require_active()
        if self._bound_connection is not None:
            raise RunStateError("nested scoped Run-state sessions are not supported")
        return RunStateSession(
            self._location,
            schema=self._schema,
            _connection=connection,
            _unit_identity=(
                str(unit_identity[0]),
                str(unit_identity[1]),
                str(unit_identity[2]),
            ),
            _execution=self._execution,
            _execution_owner=self._execution_owner,
        )

    def publication_scoped(self, connection: PostgresConnection) -> RunStateSession:
        """Bind a child facade to an owner-fenced publication transaction."""

        self._require_active()
        if self._bound_connection is not None:
            raise RunStateError("nested scoped Run-state sessions are not supported")
        if self._execution_owner is None:
            raise RunStateError("canonical Run publication requires an execution owner")
        return RunStateSession(
            self._location,
            schema=self._schema,
            _connection=connection,
            _execution=self._execution,
            _execution_owner=self._execution_owner,
        )

    def invalidate(self) -> None:
        """Permanently reject use after a borrowed transaction scope exits."""
        if self._bound_connection is None:
            raise RunStateError("only scoped Run-state sessions can be invalidated")
        self._active = False

    def _require_active(self) -> None:
        if not self._active:
            raise RunStateError("scoped Run-state session is no longer active")
