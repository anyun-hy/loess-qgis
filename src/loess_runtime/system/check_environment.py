"""Validate the Schema v2 semantic deployment environment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension.*",
    category=UserWarning,
)


def main() -> int:
    """Dispatch isolated workers or write one deployment environment report."""

    from loess_runtime.system import environment_report, model_probe

    parser = argparse.ArgumentParser(
        description="Validate Schema v2 inference deployment"
    )
    parser.add_argument("--scripts-dir")
    parser.add_argument("--asset-base-dir", default="")
    parser.add_argument("--conda-env")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--contract-worker", action="store_true")
    parser.add_argument("--batch-probe-worker", action="store_true")
    parser.add_argument("--batch-probe-set-worker", action="store_true")
    parser.add_argument("--sam3-worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--model-set-json", default="")
    parser.add_argument("--device")
    parser.add_argument("--batch-candidates", default="")
    parser.add_argument("--reserve-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.batch_probe_set_worker:
        if not args.model_set_json or not args.device or not args.batch_candidates:
            parser.error(
                "--batch-probe-set-worker requires --model-set-json, --device "
                "and --batch-candidates"
            )
        try:
            model_entries = json.loads(args.model_set_json)
            if not isinstance(model_entries, list):
                raise ValueError("model set must be a JSON list")
            candidates = [
                int(value)
                for value in args.batch_candidates.split(",")
                if value.strip()
            ]
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            parser.error(f"invalid model-set Batch probe arguments: {error}")
        import torch

        def publish_model_set_probe_event(payload: dict[str, Any]) -> None:
            sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            sys.stdout.flush()

        probe_result = model_probe.probe_torchscript_model_set_batches(
            torch,
            model_entries,
            args.device,
            candidates,
            reserve_bytes=max(0, int(args.reserve_bytes)),
            progress=publish_model_set_probe_event,
        )
        sys.stdout.write(
            json.dumps(
                {"batch_probe_set": True, **probe_result},
                separators=(",", ":"),
            )
            + "\n"
        )
        return 0 if probe_result["ok"] else 2
    if args.batch_probe_worker:
        if not args.model_path or not args.device or not args.batch_candidates:
            parser.error(
                "--batch-probe-worker requires --model-path, --device and "
                "--batch-candidates"
            )
        try:
            candidates = [
                int(value)
                for value in args.batch_candidates.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--batch-candidates must be comma-separated integers")
        import torch

        def publish_probe_event(payload: dict[str, Any]) -> None:
            sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            sys.stdout.flush()

        probe_result = model_probe.probe_torchscript_batches(
            torch,
            args.model_path,
            args.device,
            candidates,
            reserve_bytes=max(0, int(args.reserve_bytes)),
            progress=publish_probe_event,
        )
        sys.stdout.write(
            json.dumps(
                {"batch_probe": True, **probe_result},
                separators=(",", ":"),
            )
            + "\n"
        )
        return 0 if probe_result["ok"] else 2
    if args.contract_worker:
        if not args.model_path or not args.device:
            parser.error("--contract-worker requires --model-path and --device")
        import torch

        ok, message = model_probe.verify_torchscript_contract(
            torch,
            args.model_path,
            args.device,
        )
        sys.stdout.write(json.dumps({"ok": ok, "message": message}) + "\n")
        return 0
    if args.sam3_worker:
        if not args.model_path or not args.device:
            parser.error("--sam3-worker requires --model-path and --device")
        try:
            from loess_runtime.sam.sam3_refine import load_sam3

            runtime = load_sam3(args.model_path, args.device)
            del runtime
            gc.collect()
            ok, message = True, f"official SAM3 runtime loaded on {args.device}"
        except Exception as exc:
            ok = False
            message = f"official SAM3 runtime failed on {args.device}: {exc}"
        sys.stdout.write(json.dumps({"ok": ok, "message": message}) + "\n")
        return 0 if ok else 2
    if not args.scripts_dir or not args.conda_env:
        parser.error("--scripts-dir and --conda-env are required")
    try:
        scripts_dir = Path(args.scripts_dir).resolve()
        raw_asset_base = str(args.asset_base_dir or "").strip()
        asset_base_dir = (
            Path(raw_asset_base).expanduser().resolve()
            if raw_asset_base
            else scripts_dir
        )
        report = environment_report.build_environment_report(
            scripts_dir=scripts_dir,
            asset_base_dir=asset_base_dir,
            conda_env=args.conda_env,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "error",
            "config_fingerprint": "",
            "effective": {},
            "checks": [
                {
                    "id": "environment_check",
                    "status": "error",
                    "value": "failed",
                    "source": "check_environment.py",
                    "message": str(exc),
                    "fix": "inspect the detailed environment log",
                }
            ],
        }
    check_id = str(os.environ.get("LOESS_ENV_CHECK_ID") or "").strip()
    started_at = str(os.environ.get("LOESS_ENV_CHECK_STARTED_AT") or "").strip()
    if check_id:
        report["check_id"] = check_id
    if started_at:
        report["started_at"] = started_at
    serialized = json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, report_path)
    sys.stdout.write(serialized)
    return 0 if report["status"] != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
