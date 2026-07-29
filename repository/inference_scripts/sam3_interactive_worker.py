"""Persistent JSONL worker for click-driven SAM3 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import shapely
from rasterio import features
from rasterio.windows import Window, from_bounds
from shapely import affinity
from shapely.geometry import Point, mapping

from sam3_refine import load_sam3, predict_sam3_candidates, resolve_sam3_device


_WRITE_LOCK = threading.Lock()


def emit(event, **payload):
    with _WRITE_LOCK:
        print(json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")), flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _affine_parameters(transform):
    return [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]


def _window_for_request(raster, request):
    click = request.get("click_raster") or {}
    click_x = float(click["x"])
    click_y = float(click["y"])
    click_col, click_row = (~raster.transform) * (click_x, click_y)
    if not (0 <= click_col < raster.width and 0 <= click_row < raster.height):
        raise ValueError("clicked point lies outside the source raster")
    bounds = request.get("geometry_bounds") or {}
    if bounds:
        raw = from_bounds(
            float(bounds["xmin"]), float(bounds["ymin"]),
            float(bounds["xmax"]), float(bounds["ymax"]),
            transform=raster.transform,
        )
        buffer_px = int(request.get("buffer_px", 32))
        row_min = math.floor(raw.row_off) - buffer_px
        row_max = math.ceil(raw.row_off + raw.height) + buffer_px
        col_min = math.floor(raw.col_off) - buffer_px
        col_max = math.ceil(raw.col_off + raw.width) + buffer_px
    else:
        crop_size = max(64, int(request.get("crop_size_px", 512)))
        half = crop_size // 2
        row_min = math.floor(click_row) - half
        row_max = row_min + crop_size
        col_min = math.floor(click_col) - half
        col_max = col_min + crop_size
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(raster.height, row_max)
    col_max = min(raster.width, col_max)
    if row_max <= row_min or col_max <= col_min:
        raise ValueError("SAM3 crop window is empty")
    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
    transform = rasterio.windows.transform(window, raster.transform)
    local_col, local_row = (~transform) * (click_x, click_y)
    if not (0 <= local_col < window.width and 0 <= local_row < window.height):
        raise ValueError("clicked point is outside the actual SAM3 crop window")
    return window, transform, (local_col, local_row), (click_x, click_y)


def _predict(runtime, request):
    started = time.time()
    session_id = str(request["session_id"])
    with rasterio.open(request["raster"]) as raster:
        window, transform, click_local, click_map = _window_for_request(raster, request)
        patch = raster.read(window=window)
        bounds = request.get("geometry_bounds") or {}
        box = None
        if bounds:
            left, top = (~transform) * (float(bounds["xmin"]), float(bounds["ymax"]))
            right, bottom = (~transform) * (float(bounds["xmax"]), float(bounds["ymin"]))
            box = (left, top, right, bottom)
        candidates = predict_sam3_candidates(runtime, patch, click_local, box)
        if not candidates:
            raise RuntimeError("SAM3 returned no valid candidate containing the clicked point")
        selected = candidates[0]
        geometry = affinity.affine_transform(
            selected["geometry"], _affine_parameters(transform)
        )
        point = Point(*click_map)
        if geometry.is_empty or not geometry.is_valid or not geometry.covers(point):
            raise RuntimeError("SAM3 candidate failed map-coordinate validity or click containment")
        crop_bounds = rasterio.windows.bounds(window, raster.transform)
        crop_polygon = shapely.box(*crop_bounds)
        if not crop_polygon.covers(geometry):
            raise RuntimeError("SAM3 candidate extends outside its crop window")
        confidence_mean = 0.0
        confidence_std = 0.0
        confidence_path = str(request.get("confidence_mosaic") or "")
        if confidence_path and Path(confidence_path).is_file():
            with rasterio.open(confidence_path) as confidence:
                raw = from_bounds(*geometry.bounds, transform=confidence.transform)
                sample_window = raw.round_offsets().round_lengths()
                sample_window = sample_window.intersection(
                    Window(0, 0, confidence.width, confidence.height)
                )
                values = confidence.read(1, window=sample_window)
                sample_transform = rasterio.windows.transform(
                    sample_window, confidence.transform
                )
                inside = features.geometry_mask(
                    [mapping(geometry)],
                    out_shape=values.shape,
                    transform=sample_transform,
                    invert=True,
                )
                samples = values[inside & np.isfinite(values)]
                if samples.size:
                    confidence_mean = float(samples.mean())
                    confidence_std = float(samples.std())
        return {
            "session_id": session_id,
            "geometry": mapping(geometry),
            "geometry_wkt": geometry.wkt,
            "score": float(selected["score"]),
            "mask_index": int(selected["mask_index"]),
            "candidate_count": len(candidates),
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "crop_window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            },
            "crop_bounds": {
                "xmin": crop_bounds[0], "ymin": crop_bounds[1],
                "xmax": crop_bounds[2], "ymax": crop_bounds[3],
            },
            "elapsed_sec": round(time.time() - started, 3),
        }


def run_worker(checkpoint, expected_sha, device, sam_version):
    checkpoint = str(Path(checkpoint).resolve())
    actual_sha = sha256_file(checkpoint)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"SAM3 checkpoint SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    resolved_device = resolve_sam3_device(device)
    runtime = load_sam3(checkpoint, resolved_device)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-interactive")
    state = {
        "active": "",
        "cancelled": set(),
        "sessions": set(),
        "closing": False,
    }
    emit(
        "worker_ready",
        checkpoint_sha256=actual_sha,
        device=resolved_device,
        sam_version=sam_version,
        pid=os.getpid(),
    )

    def completed(future, session_id):
        if session_id in state["cancelled"] or state["closing"]:
            return
        try:
            payload = future.result()
            emit("candidate_ready", **payload)
        except Exception as exc:
            emit("failed", session_id=session_id, error=str(exc))
        finally:
            if state["active"] == session_id:
                state["active"] = ""

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            command = str(request.get("command") or "")
            session_id = str(request.get("session_id") or "")
            if command == "start_session":
                if not session_id:
                    raise ValueError("start_session requires session_id")
                if session_id in state["sessions"]:
                    raise ValueError("SAM3 session is already open")
                state["sessions"].add(session_id)
                emit("session_started", session_id=session_id)
            elif command == "predict":
                if not session_id:
                    raise ValueError("predict requires session_id")
                if session_id not in state["sessions"]:
                    raise ValueError("predict requires an open SAM3 session")
                if str(request.get("checkpoint_sha256") or "") != actual_sha:
                    raise ValueError("predict checkpoint SHA256 does not match worker")
                if str(request.get("sam_version") or "") != str(sam_version or ""):
                    raise ValueError("predict SAM version does not match worker")
                if str(request.get("device") or "") != str(resolved_device):
                    raise ValueError("predict device does not match worker")
                if state["active"]:
                    emit("failed", session_id=session_id, error="another SAM3 session is active")
                    continue
                state["active"] = session_id
                emit("started", session_id=session_id)
                future = executor.submit(_predict, runtime, request)
                future.add_done_callback(lambda value, sid=session_id: completed(value, sid))
            elif command == "cancel":
                if session_id:
                    state["cancelled"].add(session_id)
                    if state["active"] == session_id:
                        state["active"] = ""
                    emit("cancelled", session_id=session_id)
            elif command == "close_session":
                if not session_id:
                    raise ValueError("close_session requires session_id")
                if state["active"] == session_id:
                    state["cancelled"].add(session_id)
                    state["active"] = ""
                state["sessions"].discard(session_id)
                emit("session_closed", session_id=session_id)
            elif command == "shutdown":
                state["closing"] = True
                emit("worker_stopping")
                break
            else:
                emit("failed", session_id=session_id, error=f"unsupported command: {command}")
        except Exception as exc:
            emit("failed", session_id="", error=str(exc))
    executor.shutdown(wait=True, cancel_futures=True)
    emit("worker_stopped")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent interactive SAM3 JSONL worker")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sam-version", default="")
    args = parser.parse_args(argv)
    try:
        run_worker(
            args.checkpoint,
            args.checkpoint_sha256,
            args.device,
            args.sam_version,
        )
        return 0
    except Exception as exc:
        emit("worker_failed", error=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
