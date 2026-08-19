"""Class-aware small connected-component regularization for semantic masks.

The production polygonizer receives normalized 14-class probabilities and then
creates one polygon for every connected component of the hard label mask.  This
module provides the bounded raster operation that belongs between those two
steps.  It never creates nodata, gaps, or overlaps: eligible small components
are reassigned as complete pixel sets to one adjacent, larger component.

Area thresholds are expressed in physical square metres.  For EPSG:3857 the
affine pixel area is corrected by the local Web-Mercator scale at the window
centre; other projected metre CRSs use the affine determinant directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from scipy import ndimage


WEB_MERCATOR_RADIUS_M = 6378137.0
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


class SmallComponentRegularizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmallComponentPolicy:
    """Frozen decision policy for one regularization candidate."""

    thresholds_m2: Mapping[int, float]
    protected_class_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({61, 62, 71})
    )
    allow_protected_targets: bool = False
    disallowed_target_class_codes: frozenset[int] = field(default_factory=frozenset)
    compatible_target_class_codes: Mapping[int, frozenset[int]] = field(
        default_factory=dict
    )
    compatibility_bypass_below_m2: float = 0.0
    maximum_source_class_loss_fraction: float | None = None
    maximum_target_class_gain_fraction: float | None = None
    minimum_remaining_class_area_m2: float = 0.0
    hard_absorb_below_m2: float = 25.0
    maximum_mean_confidence: float | None = 0.70
    maximum_probability_drop: float | None = 0.15
    probability_weight: float = 1.0
    adjacency_weight: float = 1.0
    preserve_border_components: bool = True
    preserve_elongated_components: bool = False
    elongated_minimum_area_m2: float = 10.0
    elongated_minimum_aspect_ratio: float = 6.0
    elongated_maximum_mean_width_m: float = 3.0

    def threshold_for(self, class_code: int) -> float:
        return max(0.0, float(self.thresholds_m2.get(int(class_code), 0.0)))


@dataclass(frozen=True)
class _Component:
    component_id: int
    class_index: int
    class_code: int
    pixel_count: int
    slices: tuple[slice, slice]
    touches_border: bool


def physical_pixel_area_m2(
    transform: Affine,
    crs: CRS | str,
    *,
    height: int,
    width: int,
) -> float:
    """Return a local ground-area approximation for one raster pixel."""

    affine_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    if not math.isfinite(affine_area) or affine_area <= 0:
        raise SmallComponentRegularizationError("raster affine has no positive pixel area")
    raster_crs = CRS.from_user_input(crs)
    if raster_crs.to_epsg() == 3857:
        _center_x, center_y = transform * (float(width) / 2.0, float(height) / 2.0)
        latitude = math.atan(math.sinh(float(center_y) / WEB_MERCATOR_RADIUS_M))
        return affine_area * math.cos(latitude) ** 2
    if raster_crs.is_projected:
        linear_units = str(getattr(raster_crs, "linear_units", "") or "").lower()
        if linear_units in {"metre", "meter", "metres", "meters", "m"}:
            return affine_area
    if raster_crs.is_geographic:
        _center_x, center_y = transform * (float(width) / 2.0, float(height) / 2.0)
        lat_rad = math.radians(float(center_y))
        meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        meters_per_deg_lon = (math.pi / 180.0) * 6378137.0 * math.cos(lat_rad)
        dx_m = abs(float(transform.a)) * meters_per_deg_lon
        dy_m = abs(float(transform.e)) * meters_per_deg_lat
        return float(dx_m * dy_m)
    raise SmallComponentRegularizationError(
        f"physical pixel area requires a projected metre CRS, EPSG:3857, or geographic CRS, got {raster_crs}"
    )


def _validate_inputs(
    labels: np.ndarray,
    class_codes: Sequence[int],
    valid_mask: np.ndarray | None,
    confidence: np.ndarray | None,
    probabilities: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise SmallComponentRegularizationError("labels must be a two-dimensional array")
    if not class_codes:
        raise SmallComponentRegularizationError("class_codes cannot be empty")
    valid = (
        np.asarray(valid_mask, dtype=bool)
        if valid_mask is not None
        else np.ones(values.shape, dtype=bool)
    )
    if valid.shape != values.shape:
        raise SmallComponentRegularizationError("valid_mask shape does not match labels")
    if np.any(valid):
        selected = values[valid]
        if np.any(selected < 0) or np.any(selected >= len(class_codes)):
            raise SmallComponentRegularizationError(
                "valid labels contain a class index outside class_codes"
            )
    conf = None if confidence is None else np.asarray(confidence, dtype=np.float32)
    if conf is not None and conf.shape != values.shape:
        raise SmallComponentRegularizationError("confidence shape does not match labels")
    probs = None if probabilities is None else np.asarray(probabilities, dtype=np.float32)
    if probs is not None and probs.shape != (len(class_codes), *values.shape):
        raise SmallComponentRegularizationError(
            "probabilities must have shape [len(class_codes), H, W]"
        )
    return values.astype(np.int16, copy=False), valid, conf, probs


def _component_index(
    labels: np.ndarray,
    valid: np.ndarray,
    class_codes: Sequence[int],
) -> tuple[np.ndarray, list[_Component | None]]:
    component_map = np.zeros(labels.shape, dtype=np.int32)
    components: list[_Component | None] = [None]
    next_id = 1
    height, width = labels.shape
    for class_index, class_code in enumerate(class_codes):
        local, count = ndimage.label(
            valid & (labels == int(class_index)),
            structure=EIGHT_CONNECTED,
        )
        if count == 0:
            continue
        sizes = np.bincount(local.ravel(), minlength=count + 1)
        objects = ndimage.find_objects(local, max_label=count)
        selected = local > 0
        component_map[selected] = local[selected].astype(np.int32) + next_id - 1
        for local_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            row_slice, col_slice = slices
            touches_border = (
                int(row_slice.start) == 0
                or int(col_slice.start) == 0
                or int(row_slice.stop) == height
                or int(col_slice.stop) == width
            )
            components.append(
                _Component(
                    component_id=next_id + local_id - 1,
                    class_index=int(class_index),
                    class_code=int(class_code),
                    pixel_count=int(sizes[local_id]),
                    slices=(row_slice, col_slice),
                    touches_border=touches_border,
                )
            )
        next_id += count
    if np.any(valid & (component_map == 0)):
        raise SmallComponentRegularizationError("some valid pixels were not indexed")
    return component_map, components


def _expanded_slices(
    slices: tuple[slice, slice],
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    row_slice, col_slice = slices
    return (
        slice(max(0, int(row_slice.start) - 1), min(shape[0], int(row_slice.stop) + 1)),
        slice(max(0, int(col_slice.start) - 1), min(shape[1], int(col_slice.stop) + 1)),
    )


def _resolve_root(component_id: int, decisions: Mapping[int, int]) -> int:
    seen: set[int] = set()
    current = int(component_id)
    while current in decisions:
        if current in seen:
            raise SmallComponentRegularizationError("component reassignment contains a cycle")
        seen.add(current)
        current = int(decisions[current])
    return current


def regularize_small_components(
    labels: np.ndarray,
    *,
    class_codes: Sequence[int],
    pixel_area_m2: float,
    policy: SmallComponentPolicy,
    valid_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Absorb eligible small components into adjacent larger components.

    The returned report is deliberately scalar/count based so it can be stored
    in the run-state database without creating unbounded diagnostic geometry.
    """

    values, valid, conf, probs = _validate_inputs(
        labels, class_codes, valid_mask, confidence, probabilities
    )
    pixel_area = float(pixel_area_m2)
    if not math.isfinite(pixel_area) or pixel_area <= 0:
        raise SmallComponentRegularizationError("pixel_area_m2 must be positive")
    budget_mask = (
        valid
        if class_budget_mask is None
        else np.asarray(class_budget_mask, dtype=bool) & valid
    )
    if budget_mask.shape != values.shape:
        raise SmallComponentRegularizationError(
            "class_budget_mask shape does not match labels"
        )
    component_map, components = _component_index(values, valid, class_codes)
    decisions: dict[int, int] = {}
    decision_details: dict[int, dict[str, Any]] = {}
    kept = Counter()
    eligible_count = 0
    class_index_totals = np.bincount(
        values[budget_mask], minlength=len(class_codes)
    )
    class_pixel_totals = Counter(
        {
            int(class_code): int(class_index_totals[class_index])
            for class_index, class_code in enumerate(class_codes)
        }
    )
    scheduled_source_pixels: Counter[int] = Counter()
    scheduled_target_pixels: Counter[int] = Counter()

    ordered = sorted(
        (item for item in components[1:] if item is not None),
        key=lambda item: (item.pixel_count, item.class_code, item.component_id),
    )
    for component in ordered:
        threshold_m2 = policy.threshold_for(component.class_code)
        area_m2 = float(component.pixel_count) * pixel_area
        if threshold_m2 <= 0 or area_m2 >= threshold_m2:
            kept["outside_threshold"] += 1
            continue
        eligible_count += 1
        if component.class_code in policy.protected_class_codes:
            kept["protected_class"] += 1
            continue
        if policy.preserve_border_components and component.touches_border:
            kept["touches_processing_border"] += 1
            continue

        row_slice, col_slice = component.slices
        component_height = int(row_slice.stop) - int(row_slice.start)
        component_width = int(col_slice.stop) - int(col_slice.start)
        longest_side_pixels = max(component_height, component_width)
        shortest_side_pixels = max(1, min(component_height, component_width))
        aspect_ratio = float(longest_side_pixels) / float(shortest_side_pixels)
        longest_side_m = float(longest_side_pixels) * math.sqrt(pixel_area)
        mean_width_m = area_m2 / max(longest_side_m, math.sqrt(pixel_area))
        if (
            policy.preserve_elongated_components
            and area_m2 >= float(policy.elongated_minimum_area_m2)
            and aspect_ratio >= float(policy.elongated_minimum_aspect_ratio)
            and mean_width_m <= float(policy.elongated_maximum_mean_width_m)
        ):
            kept["elongated_component"] += 1
            continue

        expanded = _expanded_slices(component.slices, values.shape)
        local_components = component_map[expanded]
        local_component = local_components == component.component_id
        ring = ndimage.binary_dilation(
            local_component,
            structure=EIGHT_CONNECTED,
            iterations=1,
        ) & ~local_component
        neighbor_ids, contacts = np.unique(local_components[ring], return_counts=True)
        candidates: list[tuple[int, int]] = []
        for raw_id, raw_contact in zip(neighbor_ids.tolist(), contacts.tolist()):
            neighbor_id = int(raw_id)
            if neighbor_id <= 0 or neighbor_id == component.component_id:
                continue
            neighbor = components[neighbor_id]
            if neighbor is None or neighbor.class_index == component.class_index:
                continue
            if (
                not policy.allow_protected_targets
                and neighbor.class_code in policy.protected_class_codes
            ):
                continue
            if neighbor.class_code in policy.disallowed_target_class_codes:
                continue
            compatible_targets = policy.compatible_target_class_codes.get(
                component.class_code
            )
            if (
                compatible_targets
                and area_m2 >= float(policy.compatibility_bypass_below_m2)
                and neighbor.class_code not in compatible_targets
            ):
                continue
            if neighbor.pixel_count < component.pixel_count:
                continue
            if (
                neighbor.pixel_count == component.pixel_count
                and neighbor.component_id > component.component_id
            ):
                continue
            candidates.append((neighbor_id, int(raw_contact)))
        if not candidates:
            kept["no_larger_neighbor"] += 1
            continue

        source_region = component_map[component.slices] == component.component_id
        mean_confidence = (
            float(np.mean(conf[component.slices][source_region]))
            if conf is not None and np.any(source_region)
            else None
        )
        if (
            probs is None
            and policy.maximum_mean_confidence is not None
            and area_m2 >= float(policy.hard_absorb_below_m2)
            and mean_confidence is not None
            and mean_confidence > float(policy.maximum_mean_confidence)
        ):
            kept["high_confidence"] += 1
            continue

        contact_by_class: dict[int, int] = defaultdict(int)
        candidate_ids_by_class: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for neighbor_id, contact in candidates:
            neighbor = components[neighbor_id]
            assert neighbor is not None
            contact_by_class[neighbor.class_index] += contact
            candidate_ids_by_class[neighbor.class_index].append((neighbor_id, contact))
        total_contact = max(1, sum(contact_by_class.values()))
        probability_means: dict[int, float] = {}
        original_probability = None
        if probs is not None:
            original_probability = float(
                np.mean(probs[component.class_index][component.slices][source_region])
            )
            for candidate_class in contact_by_class:
                probability_means[candidate_class] = float(
                    np.mean(probs[candidate_class][component.slices][source_region])
                )

        def class_score(candidate_class: int) -> tuple[float, int, int]:
            adjacency = float(contact_by_class[candidate_class]) / float(total_contact)
            probability = probability_means.get(candidate_class, 0.0)
            return (
                float(policy.adjacency_weight) * adjacency
                + float(policy.probability_weight) * probability,
                int(contact_by_class[candidate_class]),
                -int(class_codes[candidate_class]),
            )

        target_class = max(contact_by_class, key=class_score)
        target_probability = probability_means.get(target_class)
        if (
            probs is not None
            and policy.maximum_probability_drop is not None
            and area_m2 >= float(policy.hard_absorb_below_m2)
            and original_probability is not None
            and target_probability is not None
            and target_probability
            < original_probability - float(policy.maximum_probability_drop)
        ):
            kept["probability_drop"] += 1
            continue
        component_budget_pixel_count = int(
            np.count_nonzero(
                source_region & budget_mask[component.slices]
            )
        )
        source_loss_after = scheduled_source_pixels[component.class_code]
        source_loss_after += component_budget_pixel_count
        maximum_loss_fraction = policy.maximum_source_class_loss_fraction
        if (
            maximum_loss_fraction is not None
            and component_budget_pixel_count > 0
            and source_loss_after
            > float(class_pixel_totals[component.class_code])
            * max(0.0, float(maximum_loss_fraction))
        ):
            kept["source_class_loss_budget"] += 1
            continue
        remaining_source_area_m2 = (
            float(class_pixel_totals[component.class_code] - source_loss_after)
            * pixel_area
        )
        if (
            component_budget_pixel_count > 0
            and remaining_source_area_m2
            < float(policy.minimum_remaining_class_area_m2)
        ):
            kept["source_class_minimum_area"] += 1
            continue
        target_code = int(class_codes[target_class])
        target_gain_after = scheduled_target_pixels[target_code]
        target_gain_after += component_budget_pixel_count
        maximum_gain_fraction = policy.maximum_target_class_gain_fraction
        if (
            maximum_gain_fraction is not None
            and component_budget_pixel_count > 0
            and target_gain_after
            > float(class_pixel_totals[target_code])
            * max(0.0, float(maximum_gain_fraction))
        ):
            kept["target_class_gain_budget"] += 1
            continue
        target_id = max(
            candidate_ids_by_class[target_class],
            key=lambda pair: (
                pair[1],
                components[pair[0]].pixel_count if components[pair[0]] else 0,
                -pair[0],
            ),
        )[0]
        decisions[component.component_id] = int(target_id)
        scheduled_source_pixels[component.class_code] = source_loss_after
        scheduled_target_pixels[target_code] = target_gain_after
        decision_details[component.component_id] = {
            "source_code": component.class_code,
            "target_code": int(class_codes[target_class]),
            "pixel_count": component.pixel_count,
            "area_m2": area_m2,
            "mean_confidence": mean_confidence,
            "original_probability": original_probability,
            "target_probability": target_probability,
        }

    output = values.copy()
    changed_by_pair: Counter[tuple[int, int]] = Counter()
    changed_component_count = 0
    changed_pixel_count = 0
    changed_area_m2 = 0.0
    for component_id, detail in decision_details.items():
        root_id = _resolve_root(component_id, decisions)
        root = components[root_id]
        component = components[component_id]
        if root is None or component is None or root.class_index == component.class_index:
            continue
        selected = component_map[component.slices] == component_id
        output_view = output[component.slices]
        output_view[selected] = int(root.class_index)
        changed_component_count += 1
        changed_pixel_count += component.pixel_count
        changed_area_m2 += component.pixel_count * pixel_area
        changed_by_pair[(component.class_code, root.class_code)] += 1

    report = {
        "schema_version": 1,
        "pixel_area_m2": pixel_area,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "class_budget_pixel_count": int(np.count_nonzero(budget_mask)),
        "component_count_before": len(components) - 1,
        "eligible_component_count": int(eligible_count),
        "changed_component_count": int(changed_component_count),
        "changed_pixel_count": int(changed_pixel_count),
        "changed_area_m2": float(changed_area_m2),
        "kept_reason_counts": dict(sorted(kept.items())),
        "changed_pair_counts": {
            f"{source}->{target}": int(count)
            for (source, target), count in sorted(changed_by_pair.items())
        },
        "protected_class_codes": sorted(int(value) for value in policy.protected_class_codes),
        "allow_protected_targets": bool(policy.allow_protected_targets),
        "disallowed_target_class_codes": sorted(
            int(value) for value in policy.disallowed_target_class_codes
        ),
        "compatible_target_class_codes": {
            str(int(source)): sorted(int(target) for target in targets)
            for source, targets in sorted(policy.compatible_target_class_codes.items())
        },
        "compatibility_bypass_below_m2": float(
            policy.compatibility_bypass_below_m2
        ),
        "maximum_source_class_loss_fraction": (
            None
            if policy.maximum_source_class_loss_fraction is None
            else float(policy.maximum_source_class_loss_fraction)
        ),
        "maximum_target_class_gain_fraction": (
            None
            if policy.maximum_target_class_gain_fraction is None
            else float(policy.maximum_target_class_gain_fraction)
        ),
        "minimum_remaining_class_area_m2": float(
            policy.minimum_remaining_class_area_m2
        ),
        "thresholds_m2": {
            str(int(code)): float(value)
            for code, value in sorted(policy.thresholds_m2.items())
        },
        "hard_absorb_below_m2": float(policy.hard_absorb_below_m2),
        "maximum_mean_confidence": policy.maximum_mean_confidence,
        "maximum_probability_drop": policy.maximum_probability_drop,
        "preserve_elongated_components": bool(
            policy.preserve_elongated_components
        ),
        "elongated_minimum_area_m2": float(policy.elongated_minimum_area_m2),
        "elongated_minimum_aspect_ratio": float(
            policy.elongated_minimum_aspect_ratio
        ),
        "elongated_maximum_mean_width_m": float(
            policy.elongated_maximum_mean_width_m
        ),
    }
    return output, report
