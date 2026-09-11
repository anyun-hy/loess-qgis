from pathlib import Path

from labeling_tool.core.memory_admission import (
    AdaptiveMemoryAdmissionController,
    GIB,
    MemoryPressureSample,
    read_linux_memory_pressure,
)


def _sample(*, some=0.0, full=0.0, available_gib=80, swap_used=0.0):
    swap_total = 16 * GIB
    return MemoryPressureSample(
        supported=True,
        total_bytes=100 * GIB,
        available_bytes=int(available_gib * GIB),
        swap_total_bytes=swap_total,
        swap_free_bytes=int(swap_total * (1.0 - swap_used)),
        some_avg10=float(some),
        full_avg10=float(full),
        pressure_source_count=2,
    )


def test_linux_sampler_uses_strongest_pressure_from_cgroup_ancestry(tmp_path):
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    leaf = (
        cgroup
        / "user.slice/user-1000.slice/user@1000.service/app.slice/qgis.scope"
    )
    parent = cgroup / "user.slice/user-1000.slice/user@1000.service"
    (proc / "pressure").mkdir(parents=True)
    (proc / "self").mkdir(parents=True)
    leaf.mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal:       102400 kB\n"
        "MemAvailable:    51200 kB\n"
        "SwapTotal:       16384 kB\n"
        "SwapFree:         8192 kB\n",
        encoding="utf-8",
    )
    (proc / "pressure" / "memory").write_text(
        "some avg10=4.00 avg60=2.00 avg300=1.00 total=1\n"
        "full avg10=0.50 avg60=0.20 avg300=0.10 total=1\n",
        encoding="utf-8",
    )
    (proc / "self" / "cgroup").write_text(
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/qgis.scope\n",
        encoding="utf-8",
    )
    (leaf / "memory.current").write_text("21474836480\n", encoding="utf-8")
    (leaf / "memory.stat").write_text(
        "anon 12884901888\nfile 8589934592\n",
        encoding="utf-8",
    )
    (leaf / "memory.pressure").write_text(
        "some avg10=12.00 avg60=8.00 avg300=2.00 total=2\n"
        "full avg10=2.00 avg60=1.00 avg300=0.20 total=2\n",
        encoding="utf-8",
    )
    (parent / "memory.pressure").write_text(
        "some avg10=63.19 avg60=20.00 avg300=5.00 total=3\n"
        "full avg10=3.00 avg60=1.00 avg300=0.30 total=3\n",
        encoding="utf-8",
    )

    sample = read_linux_memory_pressure(proc_root=proc, cgroup_root=cgroup)

    assert sample.supported is True
    assert sample.total_bytes == 102400 * 1024
    assert sample.available_bytes == 51200 * 1024
    assert sample.cgroup_current_bytes == 21474836480
    assert sample.cgroup_anon_bytes == 12884901888
    assert sample.cgroup_file_bytes == 8589934592
    assert sample.some_avg10 == 63.19
    assert sample.full_avg10 == 3.0
    assert sample.pressure_source_count >= 3


def test_controller_warms_up_then_grows_one_slot_at_a_time():
    controller = AdaptiveMemoryAdmissionController()
    stable = _sample()

    first = controller.decide(
        static_limit=16,
        active_slots=0,
        package_active=True,
        now=0.0,
        sample=stable,
    )
    before_growth = controller.decide(
        static_limit=16,
        active_slots=4,
        package_active=True,
        now=4.9,
        sample=stable,
    )
    controller.observe_worker_peak(2 * GIB)
    after_growth = controller.decide(
        static_limit=16,
        active_slots=4,
        package_active=True,
        now=5.0,
        sample=stable,
    )

    assert first.geometry_slot_limit == 4
    assert before_growth.geometry_slot_limit == 4
    assert after_growth.geometry_slot_limit == 5
    assert after_growth.reason == "stable_growth"


def test_controller_halves_early_pressure_and_quarters_severe_pressure():
    controller = AdaptiveMemoryAdmissionController(
        {
            "initial_geometry_slots_with_package": 16,
            "initial_geometry_slots_without_package": 16,
        }
    )
    controller.decide(
        static_limit=16,
        active_slots=16,
        package_active=True,
        now=0.0,
        sample=_sample(),
    )

    pressured = controller.decide(
        static_limit=16,
        active_slots=16,
        package_active=True,
        now=1.0,
        sample=_sample(some=8.0),
    )
    severe = controller.decide(
        static_limit=16,
        active_slots=8,
        package_active=True,
        now=2.0,
        sample=_sample(some=25.0),
    )

    assert pressured.geometry_slot_limit == 8
    assert pressured.pause_new_work is True
    assert pressured.shed_active_work is False
    assert pressured.reason == "memory_pressure"
    assert severe.geometry_slot_limit == 2
    assert severe.pause_new_work is True
    assert severe.shed_active_work is True
    assert severe.reason == "severe_memory_pressure"


def test_repeated_pressure_sample_drains_queue_without_cascading_shed():
    controller = AdaptiveMemoryAdmissionController({"initial_geometry_slots_with_package": 16})
    for now, active in [(0.0, 16), (0.2, 14), (0.6, 8), (1.0, 4)]:
        result = controller.decide(static_limit=16, active_slots=active,
                                   package_active=True, now=now, sample=_sample(full=1.08))
        assert result.geometry_slot_limit == 8
        assert result.pause_new_work
        assert not result.shed_active_work
    later = controller.decide(static_limit=16, active_slots=8, package_active=True,
                              now=5.0, sample=_sample(full=1.08))
    assert later.geometry_slot_limit == 4


def test_severe_pressure_can_escalate_once_but_not_on_every_callback():
    controller = AdaptiveMemoryAdmissionController({"initial_geometry_slots_with_package": 16})
    first = controller.decide(static_limit=16, active_slots=16, package_active=True,
                              now=0, sample=_sample(available_gib=5))
    assert first.geometry_slot_limit == 4
    for now, active in [(0.1, 12), (0.4, 5), (0.8, 4)]:
        result = controller.decide(static_limit=16, active_slots=active, package_active=True,
                                   now=now, sample=_sample(available_gib=5))
        assert result.geometry_slot_limit == 4


def test_cached_sample_cannot_reduce_again_even_after_cooldown():
    controller = AdaptiveMemoryAdmissionController(
        {"sample_interval_sec": 60, "initial_geometry_slots_with_package": 16},
        sampler=lambda: _sample(full=1.08),
    )
    first = controller.decide(static_limit=16, active_slots=16, package_active=True, now=0)
    second = controller.decide(static_limit=16, active_slots=8, package_active=True, now=10)
    assert first.geometry_slot_limit == second.geometry_slot_limit == 8


def test_controller_recovers_only_after_a_new_stable_growth_window():
    controller = AdaptiveMemoryAdmissionController(
        {"initial_geometry_slots_with_package": 8}
    )
    pressure = controller.decide(
        static_limit=16,
        active_slots=8,
        package_active=True,
        now=0.0,
        sample=_sample(some=10.0),
    )
    recovered = controller.decide(
        static_limit=16,
        active_slots=4,
        package_active=True,
        now=1.0,
        sample=_sample(),
    )
    controller.observe_worker_peak(2 * GIB)
    grown = controller.decide(
        static_limit=16,
        active_slots=4,
        package_active=True,
        now=5.0,
        sample=_sample(),
    )

    assert pressure.geometry_slot_limit == 4
    assert recovered.geometry_slot_limit == 4
    assert recovered.pause_new_work is False
    assert grown.geometry_slot_limit == 5


def test_controller_does_not_grow_while_current_allowance_is_idle():
    controller = AdaptiveMemoryAdmissionController()
    stable = _sample()
    controller.decide(
        static_limit=16,
        active_slots=0,
        package_active=True,
        now=0.0,
        sample=stable,
    )
    controller.observe_worker_peak(2 * GIB)

    idle = controller.decide(
        static_limit=16,
        active_slots=0,
        package_active=True,
        now=60.0,
        sample=stable,
    )

    assert idle.geometry_slot_limit == 4
    assert idle.reason == "stable_warmup"


def test_unavailable_sensor_preserves_static_cross_platform_fallback():
    controller = AdaptiveMemoryAdmissionController()

    decision = controller.decide(
        static_limit=12,
        active_slots=0,
        package_active=False,
        now=0.0,
        sample=MemoryPressureSample(),
    )

    assert decision.geometry_slot_limit == 12
    assert decision.pause_new_work is False
    assert decision.reason == "pressure_sensor_unavailable_static_fallback"


def test_zero_available_memory_remains_a_supported_severe_sample(tmp_path):
    proc = tmp_path / "proc"
    (proc / "pressure").mkdir(parents=True)
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal:       102400 kB\nMemAvailable:        0 kB\n",
        encoding="utf-8",
    )
    (proc / "self" / "cgroup").write_text("", encoding="utf-8")

    sample = read_linux_memory_pressure(
        proc_root=proc,
        cgroup_root=tmp_path / "missing-cgroup",
    )

    assert sample.supported is True
    assert sample.available_bytes == 0
    decision = AdaptiveMemoryAdmissionController().decide(
        static_limit=16,
        active_slots=8,
        package_active=True,
        now=0.0,
        sample=sample,
    )
    assert decision.reason == "severe_memory_pressure"
    assert decision.pause_new_work is True


def test_successful_worker_peaks_raise_the_conservative_estimate():
    controller = AdaptiveMemoryAdmissionController(
        {"default_worker_peak_bytes": GIB}
    )
    controller.observe_worker_peak(2 * GIB)

    decision = controller.decide(
        static_limit=8,
        active_slots=0,
        package_active=True,
        now=0.0,
        sample=_sample(),
    )

    assert decision.worker_peak_estimate_bytes == int(2 * GIB * 1.20)
