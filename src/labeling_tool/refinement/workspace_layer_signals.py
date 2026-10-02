"""Own QGIS signal connections for the active class-workspace layers.

Public entry: :class:`WorkspaceLayerSignals`.  It accepts only the QGIS layer
objects that it connects, and owns their connection lifecycle.  The dialog
continues to own class/workspace state, edit tracking, and UI updates.
"""

from __future__ import annotations

from qgis.core import QgsLayerTreeLayer, QgsVectorLayer
from qgis.PyQt.QtCore import QMetaObject, QObject, pyqtBoundSignal, pyqtSignal


class WorkspaceLayerSignals(QObject):
    """Forward workspace-layer events and own all associated connections.

    The object stores native ``QMetaObject.Connection`` handles.  Therefore
    cleanup can disconnect a live layer after it has left ``QgsProject`` and
    does not need to access a destroyed source object's bound signal.
    Use it on the GUI thread and call ``cleanup`` before destroying its owner.
    """

    editing_started = pyqtSignal(int)
    editing_stopped = pyqtSignal(int)
    before_commit = pyqtSignal(int, object)
    features_committed = pyqtSignal(int, object)
    selection_changed = pyqtSignal(int)
    edit_changed = pyqtSignal(int)
    visibility_changed = pyqtSignal(int)
    current_layer_changed = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._layers: dict[
            str, tuple[int, QgsVectorLayer, list[QMetaObject.Connection]]
        ] = {}
        self._tree_connections: dict[
            str, tuple[QgsLayerTreeLayer, QMetaObject.Connection]
        ] = {}
        self._current_layer_signal: pyqtBoundSignal | None = None
        self._current_layer_connection: QMetaObject.Connection | None = None

    def bind_layer(
        self,
        class_code: int,
        layer: QgsVectorLayer,
        tree_layer: QgsLayerTreeLayer | None = None,
    ) -> None:
        """Bind one class layer, optionally adding its current tree node.

        Rebinding the same class/layer is idempotent.  Supplying a tree node
        later adds its visibility connection; replacing or removing a node
        detaches the prior node before the new one is connected.
        """
        class_code = int(class_code)
        layer_id = layer.id()
        existing = self._layers.get(layer_id)
        if existing is not None and (
            existing[0] != class_code or existing[1] is not layer
        ):
            self._clear_layer(layer_id)
            existing = None
        if existing is None:
            connections: list[QMetaObject.Connection] = []
            self._layers[layer_id] = (class_code, layer, connections)
            try:
                self._connect_layer_signals(class_code, layer, connections)
            except Exception:
                self._clear_layer(layer_id)
                raise

        self._bind_tree_layer(layer_id, class_code, tree_layer)

    def clear_layers(self) -> None:
        """Disconnect layer, undo-stack, and tree-node signals only."""
        for layer_id in tuple(self._layers):
            self._clear_layer(layer_id)

    def connect_current_layer(self, signal: pyqtBoundSignal | None) -> None:
        """Forward the iface ``currentLayerChanged`` signal when available."""
        if signal == self._current_layer_signal:
            return
        if self._current_layer_connection is not None:
            self._disconnect_current_layer()
        if signal is None:
            return
        try:
            self._current_layer_connection = signal.connect(
                self.current_layer_changed.emit
            )
        except (TypeError, RuntimeError):
            return
        self._current_layer_signal = signal

    def cleanup(self) -> None:
        """Disconnect every owned connection; repeated calls are safe."""
        self.clear_layers()
        self._disconnect_current_layer()

    def _connect_layer_signals(
        self,
        class_code: int,
        layer: QgsVectorLayer,
        connections: list[QMetaObject.Connection],
    ) -> None:
        connections.append(
            layer.editingStarted.connect(
                lambda c=class_code: self.editing_started.emit(c)
            )
        )
        connections.append(
            layer.afterCommitChanges.connect(
                lambda c=class_code: self.editing_stopped.emit(c)
            )
        )
        connections.append(
            layer.editingStopped.connect(
                lambda c=class_code: self.editing_stopped.emit(c)
            )
        )
        connections.append(
            layer.beforeCommitChanges.connect(
                lambda *_args, c=class_code, current=layer: self.before_commit.emit(
                    c, current
                )
            )
        )
        connections.append(
            layer.committedFeaturesAdded.connect(
                lambda _layer_id, features, c=class_code: self.features_committed.emit(
                    c, features
                )
            )
        )
        connections.append(
            layer.selectionChanged.connect(
                lambda *_args, c=class_code: self.selection_changed.emit(c)
            )
        )
        for signal_name in (
            "geometryChanged",
            "featureAdded",
            "featureDeleted",
            "attributeValueChanged",
        ):
            signal = getattr(layer, signal_name, None)
            if signal is not None:
                connections.append(
                    signal.connect(
                        lambda *_args, c=class_code: self.edit_changed.emit(c)
                    )
                )
        undo_stack = layer.undoStack()
        connections.append(
            undo_stack.indexChanged.connect(
                lambda *_args, c=class_code: self.edit_changed.emit(c)
            )
        )

    def _bind_tree_layer(
        self,
        layer_id: str,
        class_code: int,
        tree_layer: QgsLayerTreeLayer | None,
    ) -> None:
        current = self._tree_connections.get(layer_id)
        if current is not None and current[0] is not tree_layer:
            QObject.disconnect(current[1])
            self._tree_connections.pop(layer_id, None)
            current = None
        if tree_layer is None or current is not None:
            return
        connection = tree_layer.visibilityChanged.connect(
            lambda _node=None, c=class_code: self.visibility_changed.emit(c)
        )
        self._tree_connections[layer_id] = (tree_layer, connection)

    def _clear_layer(self, layer_id: str) -> None:
        binding = self._layers.pop(layer_id, None)
        if binding is not None:
            for connection in binding[2]:
                QObject.disconnect(connection)
        tree_connection = self._tree_connections.pop(layer_id, None)
        if tree_connection is not None:
            QObject.disconnect(tree_connection[1])

    def _disconnect_current_layer(self) -> None:
        if self._current_layer_connection is not None:
            QObject.disconnect(self._current_layer_connection)
        self._current_layer_signal = None
        self._current_layer_connection = None
