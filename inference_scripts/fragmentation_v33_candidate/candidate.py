"""V3.3 raster executor for the approved configurable production policy.

The executor consumes one frozen V3-cleaned label/probability window and never
imports or mutates production V3.  A proposal always consumes one complete,
closed, dynamic source component.  Scenario routing is fixed as:

1. external/invalid boundary -> reject;
2. one distinct surrounding class -> direct enclosure absorption;
3. multiple classes with an eligible same-class bridge -> bridge proposal(s);
4. otherwise -> one rarity-selected multi-neighbour proposal.

All generated scenarios enter one incremental adjudication pass using the
approved V3.3 conflict order.  Probability and contact length are audit/rank
evidence only, never semantic rejection gates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fragmentation_policy import load_policy, policy_sha256 as config_sha256
from fragmentation_policy.loader import policy_snapshot as config_snapshot
from . import engine as _b


V33_POLICY_ID = "fragmentation_v33_configurable_absorption_v1"
V33_POLICY_VERSION = "v33_production_20260826"
V33_ADJUDICATION_MODE = "approved_fragment_reduction_rarity_incremental_v1"


class V33CandidateError(RuntimeError):
    """Raised when V3.3 input, policy, or a hard contract is invalid."""


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_policy(document: Mapping[str, Any] | None = None) -> _b.CandidatePolicy:
    """Translate the strict V3.3 document into the proven topology primitives."""

    policy = config_snapshot(document or load_policy())
    rows: dict[int, _b.ClassPolicy] = {}
    compatible: dict[int, frozenset[int]] = {}
    target_codes = {
        int(code) for code, row in policy["classes"].items()
        if row["target_growth"] == "allow"
    }
    bridge_limits = policy["constraints"]["budgets"]["bridge_limits"]
    for raw_code, row in policy["classes"].items():
        code = int(raw_code)
        source_allowed = row["source_absorption"] == "allow"
        bridge = bridge_limits[raw_code]
        rows[code] = _b.ClassPolicy(
            dynamic_fragmentation_m2=float(row["fragment_max_m2"]),
            ordinary_protected=not source_allowed,
            enclosed_island_max_m2=float(row["fragment_max_m2"] if source_allowed else 0.0),
            allow_same_class_bridge=bridge["allow"] == "allow",
            bridge_max_edge_distance_m=float(bridge["max_edge_distance_m"]),
            bridge_max_new_footprint_m2=float(bridge["max_new_footprint_m2"]),
            minimum_target_probability_mean=0.0,
            maximum_current_minus_target_probability_mean=1.0,
            minimum_target_probability_p10=0.0,
        )
        if source_allowed:
            compatible[code] = frozenset(target_codes - {code})
    budgets = policy["constraints"]["budgets"]
    return _b.CandidatePolicy(
        class_policies=MappingProxyType(rows),
        semantic_compatible_targets=MappingProxyType(compatible),
        protected_source_codes=frozenset(int(value) for value in budgets["protected_source_codes"]),
        maximum_source_loss_fraction=float(budgets["source_loss_fraction"]),
        maximum_target_gain_fraction=float(budgets["target_gain_fraction"]),
        protected_bridge_gain_fraction=float(budgets["protected_bridge_gain_fraction"]),
        island_maximum_mean_confidence=1.0,
        audit_proposal_limit=2**31 - 1,
        policy_id=V33_POLICY_ID,
        policy_version=V33_POLICY_VERSION,
    )


def policy_snapshot(document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = config_snapshot(document or load_policy())
    return {
        "policy_id": V33_POLICY_ID,
        "policy_version": V33_POLICY_VERSION,
        "config_policy": policy,
        "config_policy_sha256": config_sha256(policy),
        "executor_contract": {
            "baseline": "one_frozen_v3_cleaned_window",
            "source_scope": "complete_closed_dynamic_component_only",
            "scenario_order": [
                "invalid_or_ambiguous",
                "unique_enclosure",
                "same_class_bridge",
                "multi_neighbour",
            ],
            "probability": "audit_and_rank_only",
            "contact_length": "audit_only",
            "adjudication": "one_incremental_pass_all_scenarios",
            "publication": "core_owner_only",
        },
    }


def policy_snapshot_sha256(document: Mapping[str, Any] | None = None) -> str:
    return _sha_json(policy_snapshot(document))


def executor_snapshot_sha256() -> str:
    """Hash every repository file that defines V3.3 decisions."""

    scripts_root = Path(__file__).resolve().parents[1]
    paths = {
        Path(__file__).resolve(),
        Path(__file__).with_name("engine.py"),
        scripts_root / "fragmentation_v33_work_package.py",
        scripts_root / "fragmentation_global_connectivity.py",
        scripts_root / "authoritative_raster.py",
        scripts_root / "partition_mosaic.py",
    }
    policy_root = scripts_root / "fragmentation_policy"
    paths.update(policy_root.rglob("*.py"))
    paths.update(policy_root.rglob("*.yaml"))
    records = []
    for path in sorted(paths):
        records.append(
            {
                "path": str(path.relative_to(scripts_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return _sha_json({"files": records})


def _probability_evidence(
    footprint: np.ndarray, labels: np.ndarray, target_index: int, probabilities: np.ndarray,
) -> dict[str, float]:
    rows, cols = footprint[:, 0], footprint[:, 1]
    target = probabilities[target_index, rows, cols].astype(np.float64, copy=False)
    current = probabilities[labels[rows, cols], rows, cols].astype(np.float64, copy=False)
    rank = max(0, int(math.ceil(0.10 * len(target))) - 1)
    return {
        "mean_target_probability": float(np.mean(target)),
        "mean_current_minus_target": float(np.mean(current - target)),
        "p10_target_probability": float(np.partition(target, rank)[rank]),
    }


def _make_proposal(
    *, kind: str, source: _b._Component, target_index: int,
    target_component_ids: Sequence[int], labels: np.ndarray, valid: np.ndarray,
    probabilities: np.ndarray, class_codes: Sequence[int], component_map: np.ndarray,
    components_by_id: Mapping[int, _b._Component], policy: _b.CandidatePolicy,
    pixel_area_m2: float, rarity_share: float, edge_distance_m: float | None = None,
    path_length_m: float | None = None, extra_evidence: Mapping[str, float] | None = None,
) -> _b._Proposal:
    footprint = np.ascontiguousarray(source.pixels, dtype=np.int32)
    rows, cols = footprint[:, 0], footprint[:, 1]
    if not np.all(valid[rows, cols]) or np.any(labels[rows, cols] == target_index):
        raise V33CandidateError("proposal footprint must be valid and change every source pixel")
    if set(int(value) for value in component_map[rows, cols]) != {source.component_id}:
        raise V33CandidateError("V3.3 proposal must consume exactly one complete source component")
    ids = tuple(sorted(set(int(value) for value in target_component_ids)))
    if not ids:
        raise V33CandidateError("proposal must attach to at least one baseline target component")
    dynamic, components = _b._local_topology_delta(
        footprint, component_map, components_by_id, ids, policy, pixel_area_m2,
    )
    evidence = _probability_evidence(footprint, labels, target_index, probabilities)
    evidence.update({
        "target_rarity_share": float(rarity_share),
        "complete_source_component": 1.0,
    })
    if extra_evidence:
        evidence.update({str(key): float(value) for key, value in extra_evidence.items()})
    digest = _b._footprint_digest(footprint)
    target_code = int(class_codes[target_index])
    proposal_id = _sha_json({
        "kind": kind,
        "target_class_code": target_code,
        "source_component_id": source.component_id,
        "target_component_ids": ids,
        "footprint_sha256": digest,
    })
    return _b._Proposal(
        kind=kind,
        target_index=int(target_index),
        target_code=target_code,
        footprint=footprint,
        source_indices=(int(source.class_index),),
        source_codes=(int(source.class_code),),
        source_component_ids=(int(source.component_id),),
        baseline_target_component_ids=ids,
        dynamic_reduction=int(dynamic),
        component_reduction=int(components),
        probability_support=float(evidence["mean_target_probability"]),
        area_m2=float(len(footprint) * pixel_area_m2),
        digest=digest,
        proposal_id=proposal_id,
        edge_distance_m=edge_distance_m,
        path_length_m=path_length_m,
        evidence=MappingProxyType(evidence),
    )


def _source_boundary(
    source: _b._Component, component_map: np.ndarray,
    components_by_id: Mapping[int, _b._Component], valid: np.ndarray,
    pixel_size_m: tuple[float, float],
) -> tuple[dict[int, tuple[int, ...]], dict[int, int], dict[int, float], bool]:
    height, width = valid.shape
    ids: dict[int, set[int]] = defaultdict(set)
    contacts: Counter[int] = Counter()
    lengths: Counter[int] = Counter()
    row_m, col_m = pixel_size_m
    for row, col in source.pixels:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(row + dr), int(col + dc)
            if rr < 0 or rr >= height or cc < 0 or cc >= width or not valid[rr, cc]:
                return {}, {}, {}, False
            component_id = int(component_map[rr, cc])
            if component_id == source.component_id:
                continue
            target = components_by_id[component_id]
            ids[target.class_code].add(component_id)
            contacts[target.class_code] += 1
            lengths[target.class_code] += col_m if dr else row_m
    return (
        {code: tuple(sorted(values)) for code, values in ids.items()},
        dict(contacts),
        {code: float(value) for code, value in lengths.items()},
        True,
    )


def _relation_allowed(
    config: Mapping[str, Any], source_code: int, target_code: int, scenario: str,
) -> bool:
    matches = [
        rule for rule in config["decision_engine"]["relation_rules"]
        if rule["source"] in ("*", str(source_code))
        and rule["target"] in ("*", str(target_code))
        and rule["scenario"] in ("*", scenario)
    ]
    if not matches:
        return False
    specificity = max(int(rule["specificity"]) for rule in matches)
    specific = [rule for rule in matches if int(rule["specificity"]) == specificity]
    priority = max(int(rule["priority"]) for rule in specific)
    effective = [rule for rule in specific if int(rule["priority"]) == priority]
    return any(rule["effect"] == "allow" for rule in effective) and not any(
        rule["effect"] == "deny" for rule in effective
    )


def _bridge_bottleneck_distance(
    component_ids: Sequence[int], components_by_id: Mapping[int, _b._Component],
    pixel_size_m: tuple[float, float], maximum_distance_m: float,
) -> float | None:
    """Return the MST bottleneck over exact component edge distances."""

    ids = tuple(sorted(set(int(value) for value in component_ids)))
    if len(ids) < 2:
        return None
    edges: list[tuple[float, int, int]] = []
    for offset, first in enumerate(ids):
        for second in ids[offset + 1:]:
            distance = _b._cell_polygon_edge_distance_m(
                components_by_id[first], components_by_id[second], pixel_size_m,
                maximum_distance_m,
            )
            if distance is not None:
                edges.append((float(distance), first, second))
    parent = {value: value for value in ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    bottleneck = 0.0
    joined = 0
    for distance, first, second in sorted(edges):
        a, b = find(first), find(second)
        if a == b:
            continue
        parent[b] = a
        bottleneck = max(bottleneck, distance)
        joined += 1
        if joined == len(ids) - 1:
            return bottleneck
    return None


def _bridge_path_length(
    source: _b._Component, target_component_ids: Sequence[int],
    component_map: np.ndarray, pixel_size_m: tuple[float, float],
) -> float:
    """Maximum shortest 4-neighbour path between target contacts through source."""

    source_cells = {(int(row), int(col)) for row, col in source.pixels}
    contacts: dict[int, set[tuple[int, int]]] = defaultdict(set)
    height, width = component_map.shape
    target_set = set(int(value) for value in target_component_ids)
    for row, col in source_cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width:
                component_id = int(component_map[rr, cc])
                if component_id in target_set:
                    contacts[component_id].add((row, col))
    row_m, col_m = pixel_size_m
    maximum = 0.0
    ids = sorted(target_set)
    for start_id in ids:
        distance: dict[tuple[int, int], float] = {}
        queue: list[tuple[float, int, int]] = []
        for row, col in contacts[start_id]:
            distance[(row, col)] = 0.0
            heapq.heappush(queue, (0.0, row, col))
        while queue:
            current, row, col = heapq.heappop(queue)
            if current != distance[(row, col)]:
                continue
            for dr, dc, step in ((-1, 0, row_m), (1, 0, row_m), (0, -1, col_m), (0, 1, col_m)):
                cell = (row + dr, col + dc)
                if cell not in source_cells:
                    continue
                candidate = current + step
                if candidate < distance.get(cell, math.inf):
                    distance[cell] = candidate
                    heapq.heappush(queue, (candidate, *cell))
        for target_id in ids:
            if target_id == start_id:
                continue
            reached = [distance[cell] for cell in contacts[target_id] if cell in distance]
            if not reached:
                raise V33CandidateError("complete source component did not connect recorded target contacts")
            maximum = max(maximum, min(reached))
    return float(maximum)


def _generate_proposals(
    baseline: np.ndarray, valid: np.ndarray, probabilities: np.ndarray,
    class_codes: Sequence[int], component_map: np.ndarray,
    components: Sequence[_b._Component], policy: _b.CandidatePolicy,
    config: Mapping[str, Any], pixel_area_m2: float,
    pixel_size_m: tuple[float, float], rejections: Counter[str],
) -> tuple[list[_b._Proposal], list[dict[str, Any]]]:
    components_by_id = {item.component_id: item for item in components}
    code_to_index = {int(code): index for index, code in enumerate(class_codes)}
    shares = config["decision_engine"]["statistics"]["raw_v31_optimization_before_share"]
    classes = config["classes"]
    bridge_limits = config["constraints"]["budgets"]["bridge_limits"]
    proposals: list[_b._Proposal] = []
    audit: list[dict[str, Any]] = []
    for source in components:
        area = float(len(source.pixels) * pixel_area_m2)
        record: dict[str, Any] = {
            "source_component_id": int(source.component_id),
            "source_class_code": int(source.class_code),
            "source_pixel_count": int(len(source.pixels)),
            "source_area_m2": area,
        }
        source_row = classes[str(source.class_code)]
        if source_row["source_absorption"] != "allow":
            rejections["protected_source"] += 1
            record["reason"] = "protected_source"
        elif source.touches_external:
            rejections["external_or_invalid_boundary"] += 1
            record["reason"] = "external_or_invalid_boundary"
        elif area >= float(source_row["fragment_max_m2"]):
            rejections["not_small_fragment"] += 1
            record["reason"] = "not_small_fragment"
        else:
            neighbours, contacts, contact_lengths, safe = _source_boundary(
                source, component_map, components_by_id, valid, pixel_size_m,
            )
            record["surrounding_class_codes"] = sorted(neighbours)
            record["contact_edges_by_class_audit_only"] = {
                str(code): int(value) for code, value in sorted(contacts.items())
            }
            record["contact_length_m_by_class_audit_only"] = {
                str(code): float(value) for code, value in sorted(contact_lengths.items())
            }
            if not safe or not neighbours:
                rejections["external_or_invalid_boundary"] += 1
                record["reason"] = "external_or_invalid_boundary"
            elif len(neighbours) == 1:
                target_code = next(iter(neighbours))
                if classes[str(target_code)]["target_growth"] != "allow" or not _relation_allowed(
                    config, source.class_code, target_code, "unique_enclosure",
                ):
                    rejections["target_growth_denied"] += 1
                    record["reason"] = "target_growth_denied"
                else:
                    proposal = _make_proposal(
                        kind="unique_enclosure", source=source,
                        target_index=code_to_index[target_code],
                        target_component_ids=neighbours[target_code], labels=baseline,
                        valid=valid, probabilities=probabilities, class_codes=class_codes,
                        component_map=component_map, components_by_id=components_by_id,
                        policy=policy, pixel_area_m2=pixel_area_m2,
                        rarity_share=float(shares[str(target_code)]),
                        extra_evidence={
                            "contact_edges_audit_only": float(contacts.get(target_code, 0)),
                            "contact_length_m_audit_only": float(contact_lengths.get(target_code, 0.0)),
                        },
                    )
                    proposals.append(proposal)
                    record.update({"reason": "generated", "scenario": "unique_enclosure", "chosen_target_class_code": target_code})
            else:
                bridge_candidates: list[_b._Proposal] = []
                bridge_audit: list[dict[str, Any]] = []
                for target_code in sorted(neighbours):
                    limit = bridge_limits[str(target_code)]
                    target_ids = neighbours[target_code]
                    check: dict[str, Any] = {"target_class_code": target_code, "target_component_ids": list(target_ids)}
                    if (
                        classes[str(target_code)]["target_growth"] != "allow"
                        or not _relation_allowed(config, source.class_code, target_code, "same_class_bridge")
                        or limit["allow"] != "allow"
                        or len(target_ids) < 2
                    ):
                        check["eligible"] = False
                        check["reason"] = "permission_or_component_count"
                    elif area > float(limit["max_new_footprint_m2"]):
                        check["eligible"] = False
                        check["reason"] = "bridge_footprint_cap"
                    else:
                        distance = _bridge_bottleneck_distance(
                            target_ids, components_by_id, pixel_size_m,
                            float(limit["max_edge_distance_m"]),
                        )
                        if distance is None or distance > float(limit["max_edge_distance_m"]):
                            check["eligible"] = False
                            check["reason"] = "bridge_distance_cap"
                        else:
                            path_length = _bridge_path_length(
                                source, target_ids, component_map, pixel_size_m,
                            )
                            proposal = _make_proposal(
                                kind="same_class_bridge", source=source,
                                target_index=code_to_index[target_code],
                                target_component_ids=target_ids, labels=baseline,
                                valid=valid, probabilities=probabilities,
                                class_codes=class_codes, component_map=component_map,
                                components_by_id=components_by_id, policy=policy,
                                pixel_area_m2=pixel_area_m2,
                                rarity_share=float(shares[str(target_code)]),
                                edge_distance_m=float(distance), path_length_m=path_length,
                                extra_evidence={
                                    "contact_edges_audit_only": float(contacts.get(target_code, 0)),
                                    "contact_length_m_audit_only": float(contact_lengths.get(target_code, 0.0)),
                                },
                            )
                            if proposal.component_reduction > 0 and proposal.dynamic_reduction >= 0:
                                bridge_candidates.append(proposal)
                                check.update({"eligible": True, "reason": "generated", "edge_distance_m": distance, "path_length_m": path_length})
                            else:
                                check["eligible"] = False
                                check["reason"] = "no_fragmentation_improvement"
                    bridge_audit.append(check)
                record["bridge_target_audit"] = bridge_audit
                if bridge_candidates:
                    proposals.extend(bridge_candidates)
                    record.update({"reason": "generated", "scenario": "same_class_bridge", "generated_target_class_codes": [item.target_code for item in bridge_candidates]})
                else:
                    legal = [
                        code for code in neighbours
                        if classes[str(code)]["target_growth"] == "allow"
                        and _relation_allowed(config, source.class_code, code, "multi_neighbour")
                    ]
                    if not legal:
                        rejections["no_legal_target"] += 1
                        record["reason"] = "no_legal_target"
                    else:
                        rows, cols = source.pixels[:, 0], source.pixels[:, 1]
                        mean_probability = {
                            code: float(np.mean(probabilities[code_to_index[code], rows, cols]))
                            for code in legal
                        }
                        target_code = min(
                            legal,
                            key=lambda code: (
                                float(shares[str(code)]),
                                -mean_probability[code],
                                int(code),
                            ),
                        )
                        proposal = _make_proposal(
                            kind="multi_neighbour", source=source,
                            target_index=code_to_index[target_code],
                            target_component_ids=neighbours[target_code], labels=baseline,
                            valid=valid, probabilities=probabilities,
                            class_codes=class_codes, component_map=component_map,
                            components_by_id=components_by_id, policy=policy,
                            pixel_area_m2=pixel_area_m2,
                            rarity_share=float(shares[str(target_code)]),
                            extra_evidence={
                                "contact_edges_audit_only": float(contacts.get(target_code, 0)),
                                "contact_length_m_audit_only": float(contact_lengths.get(target_code, 0.0)),
                            },
                        )
                        proposals.append(proposal)
                        record.update({
                            "reason": "generated",
                            "scenario": "multi_neighbour",
                            "chosen_target_class_code": target_code,
                            "target_ranking": [
                                {
                                    "class_code": code,
                                    "raw_v31_share": float(shares[str(code)]),
                                    "mean_target_probability": mean_probability[code],
                                }
                                for code in sorted(legal, key=lambda code: (float(shares[str(code)]), -mean_probability[code], code))
                            ],
                        })
        if record.get("reason") == "generated":
            audit.append(record)
    return proposals, audit


def _rank_key(proposal: _b._Proposal, config: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = {
        "dynamic_fragment_reduction": int(proposal.dynamic_reduction),
        "component_reduction": int(proposal.component_reduction),
        "target_rarity_share": float(proposal.evidence["target_rarity_share"]),
        "target_probability": float(proposal.evidence["mean_target_probability"]),
        "changed_area_m2": float(proposal.area_m2),
        "target_class_code": int(proposal.target_code),
    }
    values: list[Any] = []
    for rule in config["decision_engine"]["proposal_adjudication"]["proposal_rank"]:
        value = evidence[str(rule["field"])]
        values.append(-value if rule["order"] == "descending" else value)
    values.extend((str(proposal.digest), str(proposal.proposal_id)))
    return tuple(values)


def _boundary_metrics(
    labels: np.ndarray, valid: np.ndarray, pixel_size_m: tuple[float, float],
) -> dict[str, float | int]:
    horizontal = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    vertical = valid[:-1, :] & valid[1:, :] & (labels[:-1, :] != labels[1:, :])
    internal_edges = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    internal_m = float(np.count_nonzero(horizontal) * pixel_size_m[0] + np.count_nonzero(vertical) * pixel_size_m[1])
    outer_horizontal = int(np.count_nonzero(valid[:, 0]) + np.count_nonzero(valid[:, -1]))
    outer_vertical = int(np.count_nonzero(valid[0, :]) + np.count_nonzero(valid[-1, :]))
    invalid_horizontal = int(np.count_nonzero(valid[:, :-1] != valid[:, 1:]))
    invalid_vertical = int(np.count_nonzero(valid[:-1, :] != valid[1:, :]))
    outer_edges = outer_horizontal + outer_vertical + invalid_horizontal + invalid_vertical
    outer_m = float((outer_horizontal + invalid_horizontal) * pixel_size_m[0] + (outer_vertical + invalid_vertical) * pixel_size_m[1])
    return {
        "internal_boundary_edges": internal_edges,
        "internal_boundary_meters": internal_m,
        "total_boundary_edges": internal_edges + outer_edges,
        "total_boundary_meters": internal_m + outer_m,
    }


def _dynamic_area_m2(state: _b._IncrementalMetricState, policy: _b.CandidatePolicy, pixel_area_m2: float) -> float:
    return float(sum(
        size * pixel_area_m2
        for root, size in state.group_size.items()
        if state.active[root] and _b._is_dynamic_size(size, state.group_code[root], policy, pixel_area_m2)
    ))


def _projected_dynamic_area_m2(
    current_area_m2: float, state: _b._IncrementalMetricState,
    plan: Mapping[str, Any], proposal: _b._Proposal,
    policy: _b.CandidatePolicy, pixel_area_m2: float,
) -> float:
    source_roots = tuple(int(value) for value in plan["source_sizes"])
    target_roots = tuple(int(value) for value in plan["target_roots"])
    affected = set(source_roots) | set(target_roots)
    old = sum(
        state.group_size[root] * pixel_area_m2
        for root in affected
        if _b._is_dynamic_size(
            state.group_size[root], state.group_code[root], policy, pixel_area_m2,
        )
    )
    new = sum(
        int(size) * pixel_area_m2
        for root, size in plan["source_sizes"].items()
        if _b._is_dynamic_size(
            int(size), state.group_code[int(root)], policy, pixel_area_m2,
        )
    )
    target_size = int(plan["target_size"])
    if _b._is_dynamic_size(target_size, proposal.target_code, policy, pixel_area_m2):
        new += target_size * pixel_area_m2
    return float(current_area_m2 - old + new)


def _boundary_delta(
    labels: np.ndarray, valid: np.ndarray, footprint: np.ndarray,
    target_index: int, pixel_size_m: tuple[float, float],
) -> tuple[int, float]:
    """Exact internal-boundary delta for one connected footprint."""

    cells = {(int(row), int(col)) for row, col in footprint}
    height, width = labels.shape
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edge_delta = 0
    meter_delta = 0.0
    row_m, col_m = pixel_size_m
    for row, col in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = (row + dr, col + dc)
            rr, cc = other
            if not (0 <= rr < height and 0 <= cc < width) or not valid[rr, cc]:
                continue
            edge = tuple(sorted(((row, col), other)))
            if edge in seen:
                continue
            seen.add(edge)
            before = int(labels[row, col]) != int(labels[rr, cc])
            other_after = target_index if other in cells else int(labels[rr, cc])
            after = target_index != other_after
            delta = int(after) - int(before)
            edge_delta += delta
            meter_delta += delta * (col_m if dr else row_m)
    return edge_delta, float(meter_delta)


def _canonicalize(
    proposals: Sequence[_b._Proposal], config: Mapping[str, Any],
) -> tuple[list[_b._Proposal], Counter[str], list[dict[str, Any]]]:
    canonical, duplicates, duplicate_audit = _b._canonicalize_v31b_proposals(proposals)
    return sorted(canonical, key=lambda proposal: _rank_key(proposal, config)), duplicates, duplicate_audit


def _adjudicate(
    proposals: Sequence[_b._Proposal], labels: np.ndarray, class_codes: Sequence[int],
    valid: np.ndarray, policy: _b.CandidatePolicy, pixel_area_m2: float,
    pixel_size_m: tuple[float, float], component_map: np.ndarray,
    components: Sequence[_b._Component], class_budget_mask: np.ndarray,
    config: Mapping[str, Any],
    *,
    original_budget_totals: Mapping[int, int] | None = None,
    prior_source_loss: Mapping[int, int] | None = None,
    prior_target_gain: Mapping[int, int] | None = None,
    prior_protected_bridge_gain: Mapping[int, int] | None = None,
    immutable_mask: np.ndarray | None = None,
    dependency_lock_mask: np.ndarray | None = None,
    cumulative_budget_mode: bool = False,
) -> tuple[np.ndarray, list[_b._Proposal], Counter[str], dict[str, tuple[str, str]], list[dict[str, Any]], int]:
    budget_mask = class_budget_mask & valid
    baseline_totals = Counter(
        {int(code): int(value) for code, value in original_budget_totals.items()}
        if original_budget_totals is not None
        else Counter(int(class_codes[index]) for index in labels[budget_mask])
    )
    source_loss: Counter[int] = Counter(
        {int(code): int(value) for code, value in (prior_source_loss or {}).items()}
    )
    target_gain: Counter[int] = Counter(
        {int(code): int(value) for code, value in (prior_target_gain or {}).items()}
    )
    protected_bridge_gain: Counter[int] = Counter(
        {int(code): int(value) for code, value in (prior_protected_bridge_gain or {}).items()}
    )
    immutable = np.zeros(labels.shape, dtype=bool) if immutable_mask is None else np.asarray(immutable_mask, dtype=bool)
    dependency_lock = np.zeros(labels.shape, dtype=bool) if dependency_lock_mask is None else np.asarray(dependency_lock_mask, dtype=bool)
    if immutable.shape != labels.shape or dependency_lock.shape != labels.shape:
        raise V33CandidateError("immutable/dependency-lock masks must match labels")
    accepted: list[_b._Proposal] = []
    occupied = np.zeros(labels.shape, dtype=bool)
    result = labels.copy()
    skipped: Counter[str] = Counter()
    decisions: dict[str, tuple[str, str]] = {}
    components_by_id = {item.component_id: item for item in components}
    metric_state = _b._incremental_metric_state(components, policy, pixel_area_m2)
    baseline_dynamic_area = _dynamic_area_m2(metric_state, policy, pixel_area_m2)
    current_dynamic_area = baseline_dynamic_area
    baseline_boundary = _boundary_metrics(labels, valid, pixel_size_m)
    current_boundary = dict(baseline_boundary)
    proposal_pixel_roots: dict[tuple[int, int], int] = {}
    ordered, duplicates, duplicate_audit = _canonicalize(proposals, config)
    skipped.update(duplicates)
    removed_by_component: dict[int, set[tuple[int, int]]] = {}
    target_dependents: dict[int, list[_b._Proposal]] = {}
    accepted_by_id: dict[str, _b._Proposal] = {}
    proposal_pixel_owner: dict[tuple[int, int], str] = {}
    interactions: list[dict[str, Any]] = []
    for proposal in ordered:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        interaction: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "rank_key": list(_rank_key(proposal, config)),
            "source_component_ids": list(proposal.source_component_ids),
            "affected_accepted_proposal_ids": [],
            "target_attachment_checks": 0,
        }
        metric_plan: Mapping[str, Any] | None = None
        boundary_after: Mapping[str, float | int] | None = None
        projected_dynamic_area: float | None = None
        if np.any(immutable[rows, cols]):
            reason = "round1_immutable"
        elif np.any(dependency_lock[rows, cols]):
            reason = "round1_dependency_lock"
        elif not np.all(budget_mask[rows, cols]):
            reason = "outside_core_owner"
        elif np.any(occupied[rows, cols]):
            reason = "footprint_conflict"
        else:
            additions = Counter(int(class_codes[int(value)]) for value in labels[rows, cols])
            if any(source_loss[code] + count > baseline_totals[code] * policy.maximum_source_loss_fraction + 1e-12 for code, count in additions.items()):
                reason = "cumulative_source_budget" if cumulative_budget_mode else "source_budget"
            elif cumulative_budget_mode and target_gain[proposal.target_code] + len(rows) > baseline_totals[proposal.target_code] * policy.maximum_target_gain_fraction + 1e-12:
                reason = "cumulative_target_budget"
            elif (
                cumulative_budget_mode
                and proposal.kind == "same_class_bridge"
                and proposal.target_code in policy.protected_source_codes
                and protected_bridge_gain[proposal.target_code] + len(rows)
                > baseline_totals[proposal.target_code] * policy.protected_bridge_gain_fraction + 1e-12
            ):
                reason = "cumulative_protected_bridge_budget"
            elif (
                not cumulative_budget_mode
                and target_gain[proposal.target_code] + len(rows)
                > baseline_totals[proposal.target_code]
                * (
                    policy.protected_bridge_gain_fraction
                    if proposal.kind == "same_class_bridge" and proposal.target_code in policy.protected_source_codes
                    else policy.maximum_target_gain_fraction
                )
                + 1e-12
            ):
                reason = "target_budget"
            else:
                affected_components = tuple(sorted(set(int(value) for value in component_map[rows, cols] if value)))
                tentative_removed = {key: set(value) for key, value in removed_by_component.items()}
                for row, col, component_id in zip(rows, cols, component_map[rows, cols]):
                    tentative_removed.setdefault(int(component_id), set()).add((int(row), int(col)))
                if not _b._source_connectivity_safe_incremental(tentative_removed, affected_components, component_map, components_by_id):
                    reason = "source_connectivity"
                else:
                    metric_plan, metric_reason = _b._prospective_metric_plan(
                        metric_state, proposal, component_map, labels, result,
                        proposal_pixel_roots, policy, pixel_area_m2,
                    )
                    if metric_reason is not None:
                        reason = metric_reason
                    else:
                        edge_delta, meter_delta = _boundary_delta(
                            result, valid, proposal.footprint,
                            proposal.target_index, pixel_size_m,
                        )
                        boundary_after = {
                            "internal_boundary_edges": int(current_boundary["internal_boundary_edges"]) + edge_delta,
                            "internal_boundary_meters": float(current_boundary["internal_boundary_meters"]) + meter_delta,
                            "total_boundary_edges": int(current_boundary["total_boundary_edges"]) + edge_delta,
                            "total_boundary_meters": float(current_boundary["total_boundary_meters"]) + meter_delta,
                        }
                        if any(float(boundary_after[key]) > float(baseline_boundary[key]) + 1e-9 for key in baseline_boundary):
                            reason = "boundary_increase"
                        else:
                            projected_dynamic_area = _projected_dynamic_area_m2(
                                current_dynamic_area, metric_state, metric_plan,
                                proposal, policy, pixel_area_m2,
                            )
                            if projected_dynamic_area > baseline_dynamic_area + 1e-9:
                                reason = "dynamic_fragment_area_increase"
                            else:
                                affected = {
                                    item.proposal_id: item
                                    for component_id in affected_components
                                    for item in target_dependents.get(component_id, [])
                                }
                                checked = [*sorted(affected.values(), key=lambda item: item.proposal_id), proposal]
                                interaction["affected_accepted_proposal_ids"] = [item.proposal_id for item in checked if item is not proposal]
                                interaction["target_attachment_checks"] = len(checked)
                                saved = result[rows, cols].copy()
                                result[rows, cols] = proposal.target_index
                                transient = dict(accepted_by_id)
                                transient[proposal.proposal_id] = proposal
                                for row, col in proposal.footprint:
                                    proposal_pixel_owner[(int(row), int(col))] = proposal.proposal_id
                                target_safe = all(
                                    _b._target_attachment_safe_incremental(
                                        item, result, labels, component_map, components_by_id,
                                        transient, proposal_pixel_owner, target_dependents,
                                    )
                                    for item in checked
                                )
                                if not target_safe:
                                    result[rows, cols] = saved
                                    for row, col in proposal.footprint:
                                        proposal_pixel_owner.pop((int(row), int(col)), None)
                                    reason = "target_attachment"
                                else:
                                    reason = "selected"
                                    removed_by_component = tentative_removed
        if reason != "selected":
            skipped[reason] += 1
            decisions[proposal.proposal_id] = ("rejected", reason)
            interaction.update({"decision": "rejected", "reason": reason})
            interactions.append(interaction)
            continue
        occupied[rows, cols] = True
        accepted.append(proposal)
        decisions[proposal.proposal_id] = ("accepted", "selected")
        source_loss.update(additions)
        target_gain[proposal.target_code] += len(rows)
        if proposal.kind == "same_class_bridge" and proposal.target_code in policy.protected_source_codes:
            protected_bridge_gain[proposal.target_code] += len(rows)
        representative = _b._commit_metric_plan(metric_state, metric_plan)
        current_dynamic_area = float(projected_dynamic_area)
        current_boundary = dict(boundary_after)
        for row, col in proposal.footprint:
            proposal_pixel_roots[(int(row), int(col))] = representative
        for component_id in proposal.baseline_target_component_ids:
            target_dependents.setdefault(int(component_id), []).append(proposal)
        accepted_by_id[proposal.proposal_id] = proposal
        interaction.update({"decision": "accepted", "reason": "selected", "boundary_after": boundary_after})
        interactions.append(interaction)
    interactions.extend(duplicate_audit)
    return result, accepted, skipped, decisions, interactions, int(duplicates["duplicate_proposal"])


def _summary(
    proposal: _b._Proposal, decision: tuple[str, str], config: Mapping[str, Any],
) -> dict[str, Any]:
    rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
    return {
        "proposal_id": proposal.proposal_id,
        "kind": proposal.kind,
        "target_class_code": proposal.target_code,
        "source_class_codes": list(proposal.source_codes),
        "baseline_source_component_ids": list(proposal.source_component_ids),
        "baseline_target_component_ids": list(proposal.baseline_target_component_ids),
        "changed_pixels": int(len(proposal.footprint)),
        "area_m2": float(proposal.area_m2),
        "footprint_bbox": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
        "edge_distance_m": proposal.edge_distance_m,
        "path_length_m": proposal.path_length_m,
        "dynamic_fragment_reduction": proposal.dynamic_reduction,
        "component_reduction": proposal.component_reduction,
        "v33_rank_key": list(_rank_key(proposal, config)),
        "footprint_sha256": proposal.digest,
        "evidence": dict(proposal.evidence),
        "decision": decision[0],
        "reason": decision[1],
    }


def apply_v33_candidate(
    labels: np.ndarray, *, class_codes: Sequence[int], pixel_area_m2: float,
    pixel_size_m: tuple[float, float] | None = None,
    valid_mask: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    policy_document: Mapping[str, Any] | None = None,
    baseline_kind: str,
    full_audit: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the approved V3.3 rules to one frozen in-memory window."""

    if baseline_kind not in _b.ALLOWED_BASELINE_KINDS:
        raise V33CandidateError(f"baseline_kind must be one of {sorted(_b.ALLOWED_BASELINE_KINDS)}")
    config = config_snapshot(policy_document or load_policy())
    policy = runtime_policy(config)
    if tuple(int(code) for code in class_codes) != tuple(sorted(policy.class_policies)):
        raise V33CandidateError("class_codes must exactly match the configured V3.3 classes")
    try:
        baseline, valid, probs, _conf, sizes = _b._validate(
            labels, class_codes, valid_mask, probabilities, confidence,
            pixel_area_m2, pixel_size_m, policy,
        )
        budget = _b._class_budget_mask(class_budget_mask, valid)
        component_map, components = _b._component_index(baseline, valid, class_codes)
    except Exception as exc:
        raise V33CandidateError(str(exc)) from exc
    rejections: Counter[str] = Counter()
    proposals, generation_audit = _generate_proposals(
        baseline, valid, probs, class_codes, component_map, components, policy,
        config, pixel_area_m2, sizes, rejections,
    )
    result, accepted, skipped, decisions, interactions, duplicate_count = _adjudicate(
        proposals, baseline, class_codes, valid, policy, pixel_area_m2, sizes,
        component_map, components, budget, config,
    )
    published = baseline.copy()
    published[budget] = result[budget]
    if not np.array_equal(published[~valid], baseline[~valid]) or np.any(published[valid] < 0) or np.any(published[valid] >= len(class_codes)):
        raise V33CandidateError("candidate violated single-label or invalid-pixel preservation")
    if not _b._final_topology_holds(baseline, published, valid, class_codes, policy, pixel_area_m2, accepted, components):
        raise V33CandidateError("candidate violated final topology")
    before_boundary = _boundary_metrics(baseline, valid, sizes)
    after_boundary = _boundary_metrics(published, valid, sizes)
    if any(float(after_boundary[key]) > float(before_boundary[key]) + 1e-9 for key in before_boundary):
        raise V33CandidateError("candidate violated final boundary nonincrease")
    before_dynamic, before_components = _b._dynamic_count(baseline, valid, class_codes, policy, pixel_area_m2)
    after_dynamic, after_components = _b._dynamic_count(published, valid, class_codes, policy, pixel_area_m2)
    before_by_class = _b._per_class_metrics(baseline, valid, class_codes, policy, pixel_area_m2)
    after_by_class = _b._per_class_metrics(published, valid, class_codes, policy, pixel_area_m2)
    if sum(float(row["dynamic_fragment_area_m2"]) for row in after_by_class.values()) > sum(float(row["dynamic_fragment_area_m2"]) for row in before_by_class.values()) + 1e-9:
        raise V33CandidateError("candidate violated final dynamic-fragment-area nonincrease")
    canonical, _duplicates, duplicate_audit = _canonicalize(proposals, config)
    summaries = [
        _summary(item, decisions.get(item.proposal_id, ("generated", "unresolved")), config)
        for item in canonical
    ]
    source_loss: Counter[int] = Counter()
    target_gain: Counter[int] = Counter()
    bridge_gain: Counter[int] = Counter()
    for proposal in accepted:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        source_loss.update(int(class_codes[int(value)]) for value in baseline[rows, cols])
        target_gain[proposal.target_code] += len(rows)
        if proposal.kind == "same_class_bridge":
            bridge_gain[proposal.target_code] += len(rows)
    changed = valid & (published != baseline)
    runtime_snapshot = policy_snapshot(config)
    report: dict[str, Any] = {
        "candidate_label": "V3.3",
        "policy_id": V33_POLICY_ID,
        "policy_version": V33_POLICY_VERSION,
        "policy_snapshot": runtime_snapshot,
        "policy_snapshot_sha256": policy_snapshot_sha256(config),
        "config_policy_sha256": config_sha256(config),
        "adjudication_mode": V33_ADJUDICATION_MODE,
        "baseline_kind": baseline_kind,
        "single_pass_from_frozen_baseline": True,
        "cascade_generation": False,
        "topology_connectivity": 4,
        "single_label": True,
        "gap_pixels": 0,
        "overlap_pixels": 0,
        "outside_pixels": 0,
        "baseline_mask_sha256": hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest(),
        "output_mask_sha256": hashlib.sha256(np.ascontiguousarray(published).tobytes()).hexdigest(),
        "valid_mask_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(),
        "class_budget_mask_sha256": hashlib.sha256(np.ascontiguousarray(budget).tobytes()).hexdigest(),
        "class_codes": [int(code) for code in class_codes],
        "physical_metrics": {"pixel_area_m2": float(pixel_area_m2), "row_step_m": float(sizes[0]), "column_step_m": float(sizes[1])},
        "baseline": {"components_4_connected": before_components, "dynamic_fragments_4_connected": before_dynamic, "boundary": before_boundary},
        "result": {"components_4_connected": after_components, "dynamic_fragments_4_connected": after_dynamic, "boundary": after_boundary},
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_area_m2": float(np.count_nonzero(changed) * pixel_area_m2),
        "protected_source_loss_pixel_count": int(sum(source_loss[code] for code in policy.protected_source_codes)),
        "protected_source_retention": 1.0 if sum(source_loss[code] for code in policy.protected_source_codes) == 0 else 0.0,
        "transport_source_loss_pixel_count": int(source_loss[61] + source_loss[62]),
        "transport_overlay_obligation": config["constraints"]["transport_overlay"],
        "raw_generated": len(proposals),
        "proposals_canonical": len(canonical),
        "duplicate_proposal_count": duplicate_count,
        "proposals_accepted": len(accepted),
        "proposal_generation_reject_reason_counts": dict(sorted(rejections.items())),
        "proposal_reject_reason_counts": dict(sorted(skipped.items())),
        "generation_audit": generation_audit if full_audit else [],
        "proposal_audit": summaries if full_audit else [],
        "accepted": [_summary(item, ("accepted", "selected"), config) for item in accepted] if full_audit else [],
        "interaction_audit": interactions if full_audit else [],
        "duplicate_proposal_audit": duplicate_audit if full_audit else [],
        "full_audit": bool(full_audit),
        "audit_truncated": not full_audit,
        "final_topology_rollback": 0,
        "valid_pixel_count": int(valid.sum()),
        "per_class": {
            str(code): {
                "baseline": before_by_class[int(code)],
                "result": after_by_class[int(code)],
                "source_loss": int(source_loss[int(code)]),
                "target_gain": int(target_gain[int(code)]),
                "bridge_gain": int(bridge_gain[int(code)]),
                "net_pixel_drift": int(target_gain[int(code)] - source_loss[int(code)]),
            }
            for code in class_codes
        },
    }
    report["audit_sha256"] = _sha_json(report)
    return published, report


apply_v33 = apply_v33_candidate
