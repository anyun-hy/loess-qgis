"""Manual task state without QGIS layers, UI controls, or persistence."""

import pytest

from labeling_tool.refinement.manual_edit_state import ManualEditTask


def test_initial_selection_is_independent_of_input_and_task_changes():
    selected = [4, 5]
    task = ManualEditTask.for_modify(12, selected)
    other = ManualEditTask.for_modify(12, selected)
    selected.append(6)
    task.toggle_selected_feature(4)
    task.toggle_selected_feature(9)
    assert task.selected_feature_ids == [5, 9]
    assert task.selection_before == other.selected_feature_ids == [4, 5]
    assert other.pending_geometries == []
    assert task.class_code == task.target_code == 12
    assert task.state == "selecting"


def test_retry_discards_only_last_candidate_and_invalidates_its_preview():
    first, last = object(), object()
    task = ManualEditTask.for_add(12, [])
    task.append_candidate(first, "")
    task.append_candidate(last, "invalid ring")
    task.smoothing_preview = object()
    task.smoothing_error = "old preview"
    task.set_failed("last attempt failed")
    task.retry_candidate()
    assert task.pending_geometries == [first]
    assert task.pending_errors == [""]
    assert task.smoothing_preview is None and task.smoothing_error == ""
    assert task.error == ""
    task.retry_candidate()
    task.retry_candidate()
    assert task.pending_geometries == task.pending_errors == []


@pytest.mark.parametrize("state", ["selecting", "capturing", "capture_transition"])
def test_pause_preserves_candidates_and_resumes_a_pending_capture(state):
    task = ManualEditTask.for_modify(12, [4])
    geometry = object()
    task.append_candidate(geometry, "")
    task.state = state
    task.pause()
    assert task.state == "paused"
    task.pause()
    expected = "capturing" if state == "capture_transition" else state
    assert task.resume() == expected
    assert task.state == expected
    assert task.selected_feature_ids == [4]
    assert task.pending_geometries == [geometry]


def test_cancel_capture_and_failure_keep_pending_work():
    task = ManualEditTask.for_add(12, [])
    geometry = object()
    task.append_candidate(geometry, "")
    task.capture_cancelled()
    assert task.state == "capture_cancelled"
    task.set_failed("provider rejected save")
    assert task.state == "failed" and task.error == "provider rejected save"
    assert task.pending_geometries == [geometry]
    assert task.added_count == task.submitted_batch_count == 0
    task.begin_capture()
    assert task.state == "capturing"
    assert task.pending_geometries == [geometry]


def test_successive_add_batches_keep_saved_totals_but_reset_pending_state():
    task = ManualEditTask.for_add(12, [4])
    task.editing_started_by_task = True
    task.smoothing_enabled = True
    for code, count in ((21, 2), (12, 1), (21, 3)):
        task.target_code = code
        task.append_candidate(object(), "")
        task.set_failed("transient")
        task.record_add_batch(code, count)
        assert task.target_code == 12 and task.state == "capturing"
        assert task.pending_geometries == task.pending_errors == []
        assert task.smoothing_preview is None and task.error == ""
    assert task.added_count == 6 and task.submitted_batch_count == 3
    assert task.saved_counts == {21: 5, 12: 1}
    assert task.selection_before == [4] and task.editing_started_by_task
    assert task.smoothing_enabled


def test_modify_batch_totals_survive_selection_and_candidate_reset():
    task = ManualEditTask.for_modify(12, [4, 5])
    task.target_code = 21
    task.append_candidate(object(), "")
    task.record_modify_batch(1, 2, 1)
    assert task.state == "selecting" and task.target_code == 12
    assert task.selected_feature_ids == task.pending_geometries == []
    assert task.selection_before == [4, 5]
    task.toggle_selected_feature(8)
    task.record_modify_batch(1, 0, 0)
    assert task.submitted_batch_count == 2
    assert (task.modified_old_count, task.saved_new_count, task.deleted_old_count) == (
        2,
        2,
        1,
    )


def test_delete_task_uses_initial_count_without_sharing_the_selection_list():
    selected = [4, 5]
    task = ManualEditTask.for_delete(12, selected)
    selected.clear()
    assert task.selected_count == 2 and task.selection_before == [4, 5]
    assert task.kind == "delete" and task.state == "selecting"
