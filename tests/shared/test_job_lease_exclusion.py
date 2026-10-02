from __future__ import annotations


def _job_ids(database, run_id: str, job_type: str) -> list[int]:
    with database.session.connection() as connection:
        rows = connection.execute(
            """SELECT job_id FROM jobs
               WHERE run_id=%s AND job_type=%s ORDER BY job_id""",
            (run_id, job_type),
        ).fetchall()
    return [int(row["job_id"]) for row in rows]


def test_unit_lease_excludes_active_process_job_until_process_exit(
    postgres_database,
):
    run_id = "lease_exclusion_unit"
    postgres_database.run_streams.create_run(run_id, "a" * 64)
    postgres_database.jobs.insert_jobs(
        run_id,
        [
            {"job_type": "unit_fit", "unit_id": "unit-a"},
            {"job_type": "unit_fit", "unit_id": "unit-b"},
        ],
    )
    first_id, second_id = _job_ids(postgres_database, run_id, "unit_fit")
    first = postgres_database.jobs.lease_next_job(
        run_id,
        "worker-a",
        job_types=("unit_fit",),
        lease_seconds=120,
    )
    assert first is not None
    assert first["job_id"] == first_id
    assert postgres_database.jobs.interrupt_job(
        first["job_id"], first["lease_token"]
    )

    replacement = postgres_database.jobs.lease_next_job(
        run_id,
        "worker-b",
        job_types=("unit_fit",),
        lease_seconds=120,
        exclude_job_ids=(first_id,),
    )

    assert replacement is not None
    assert replacement["job_id"] == second_id
    assert postgres_database.jobs.interrupt_job(
        replacement["job_id"], replacement["lease_token"]
    )
    after_process_exit = postgres_database.jobs.lease_next_job(
        run_id,
        "worker-c",
        job_types=("unit_fit",),
        lease_seconds=120,
    )
    assert after_process_exit is not None
    assert after_process_exit["job_id"] == first_id


def test_v33_lease_excludes_active_process_job_until_process_exit(
    postgres_database,
):
    run_id = "lease_exclusion_v33"
    stream_id = "fusion:test"
    postgres_database.run_streams.create_run(run_id, "b" * 64)
    postgres_database.run_streams.register_streams(
        run_id,
        [{"stream_id": stream_id, "kind": "fusion", "profile_id": "test"}],
    )
    postgres_database.control_graph.insert_spatial_units(
        run_id,
        [
            {
                "unit_id": "finalize-a",
                "unit_type": "FragmentationV33Finalize",
                "owner_key": "owner-a",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "dependency_ids": [],
            },
            {
                "unit_id": "finalize-b",
                "unit_type": "FragmentationV33Finalize",
                "owner_key": "owner-b",
                "pixel_window": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
                "dependency_ids": [],
            },
        ],
    )
    postgres_database.jobs.insert_jobs(
        run_id,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": "finalize-a",
            },
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": "finalize-b",
            },
        ],
    )
    first_id, second_id = _job_ids(
        postgres_database, run_id, "fragmentation_v33"
    )
    first = postgres_database.jobs.lease_next_fragmentation_v33(
        run_id,
        "worker-a",
        lease_seconds=120,
        max_running=2,
    )
    assert first is not None
    assert first["job_id"] == first_id
    assert postgres_database.jobs.interrupt_job(
        first["job_id"], first["lease_token"]
    )

    replacement = postgres_database.jobs.lease_next_fragmentation_v33(
        run_id,
        "worker-b",
        lease_seconds=120,
        max_running=2,
        exclude_job_ids=(first_id,),
    )

    assert replacement is not None
    assert replacement["job_id"] == second_id
    assert postgres_database.jobs.interrupt_job(
        replacement["job_id"], replacement["lease_token"]
    )
    after_process_exit = postgres_database.jobs.lease_next_fragmentation_v33(
        run_id,
        "worker-c",
        lease_seconds=120,
        max_running=2,
    )
    assert after_process_exit is not None
    assert after_process_exit["job_id"] == first_id
