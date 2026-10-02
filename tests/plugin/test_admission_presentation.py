from labeling_tool.refinement.admission_presentation import (
    AdmissionSummarySnapshot,
    admission_summary_presentation,
)


def test_admission_summary_uses_only_supplied_prewrite_observations():
    presentation = admission_summary_presentation(
        AdmissionSummarySnapshot(
            target_path="/data/labels/accepted_labels.gpkg",
            final_feature_count=42,
            confirmed_class_count=14,
            unsaved_edit_count=0,
            topology_executed=True,
            topology_issue_count=2,
            allow_issues=True,
            background_stage="正在后台校验 accepted_labels",
            blocker_reason="后台校验尚未完成",
        )
    )

    assert presentation.target_path == "/data/labels/accepted_labels.gpkg"
    assert presentation.final_features.startswith("42 个面")
    assert "实际新增数量将在后台核对" in presentation.final_features
    assert presentation.class_confirmation == "14/14"
    assert presentation.unsaved_edits == "0"
    assert "问题数 2" in presentation.topology
    assert presentation.background_stage == "正在后台校验 accepted_labels"
    assert presentation.blocker == "后台校验尚未完成"
    assert "带问题入库" in presentation.integrity_note
    assert "完整性检查仍会执行" in presentation.integrity_note
    assert "准确率" not in "\n".join(
        (
            presentation.final_features,
            presentation.class_confirmation,
            presentation.unsaved_edits,
            presentation.topology,
            presentation.background_stage,
            presentation.blocker,
            presentation.accepted_result,
            presentation.integrity_note,
            presentation.next_action,
        )
    )


def test_missing_observations_never_become_zero_or_existing_library_claims():
    presentation = admission_summary_presentation(AdmissionSummarySnapshot())

    assert presentation.final_features == "未提供。"
    assert "未提供" in presentation.class_confirmation
    assert "未提供" in presentation.unsaved_edits
    assert presentation.topology == "尚未执行。"
    assert "当前窗口未记录" in presentation.accepted_result
    assert "库中" not in "\n".join(
        (
            presentation.final_features,
            presentation.class_confirmation,
            presentation.unsaved_edits,
            presentation.topology,
            presentation.background_stage,
            presentation.blocker,
            presentation.accepted_result,
            presentation.integrity_note,
            presentation.next_action,
        )
    )


def test_accepted_count_means_only_this_write_and_not_the_library_total():
    presentation = admission_summary_presentation(
        AdmissionSummarySnapshot(
            accepted_feature_count=7,
            accepted_warnings=("accepted_labels 已写入，但目录同步未确认",),
        )
    )

    assert presentation.accepted_result == "7 个面。"
    assert "总数" not in presentation.accepted_result
    assert presentation.accepted_warnings == (
        "accepted_labels 已写入，但目录同步未确认",
    )
