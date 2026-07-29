import argparse
import logging
import os
import sys
import types
import uuid
import warnings

import fiona
import numpy as np
import rasterio
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import shape, mapping, Point, Polygon
from rasterio.windows import from_bounds

from _device import resolve_device, validate_device

warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"User provided device_type of 'cuda'.*",
    category=UserWarning,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(message)s")
logger = logging.getLogger("sam3_refine")

SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "refined_id": "str",
        "parent_object_id": "str",
        "parent_part_id": "str",
        "class_code": "int",
        "class_name": "str",
        "confidence_mean": "float",
        "confidence_std": "float",
        "model_version": "str",
        "sam_version": "str",
        "source": "str",
    },
}

LAYER_NAME = "sam_refined_polygons"


def _install_cpu_edt_fallback():
    """Avoid SAM3's CUDA-only Triton import for CPU image inference."""
    if "sam3.model.edt" in sys.modules:
        return

    module = types.ModuleType("sam3.model.edt")

    def edt_triton(data):
        arrays = data.detach().cpu().numpy()
        result = np.stack([distance_transform_edt(item) for item in arrays])
        return torch.as_tensor(result, device=data.device, dtype=torch.float32)

    module.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = module


def _install_cpu_builder_compatibility(model_builder):
    """Patch SAM3 0.1.4's two constructor caches that hard-code CUDA."""
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.position_encoding import PositionEmbeddingSine

    original_position_init = PositionEmbeddingSine.__init__

    def position_init_cpu(self, *args, **kwargs):
        original_zeros = torch.zeros

        def zeros_cpu(*zero_args, **zero_kwargs):
            if str(zero_kwargs.get("device", "")).startswith("cuda"):
                zero_kwargs["device"] = "cpu"
            return original_zeros(*zero_args, **zero_kwargs)

        torch.zeros = zeros_cpu
        try:
            original_position_init(self, *args, **kwargs)
        finally:
            torch.zeros = original_zeros

    def create_position_encoding(precompute_resolution=None):
        return PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=None,
        )

    original_get_coords = TransformerDecoder._get_coords

    def get_coords_cpu(H, W, device):
        if str(device).startswith("cuda"):
            device = "cpu"
        return original_get_coords(H, W, device)

    model_builder._create_position_encoding = create_position_encoding
    PositionEmbeddingSine.__init__ = position_init_cpu
    TransformerDecoder._get_coords = staticmethod(get_coords_cpu)


def resolve_sam3_device(configured_device):
    """SAM3 officially uses CUDA; non-CUDA requests fall back to CPU."""
    resolved = resolve_device(configured_device)
    if resolved.startswith("cuda"):
        if not validate_device(resolved):
            raise RuntimeError(f"SAM3 requested {resolved}, but CUDA is unavailable")
        return resolved
    if resolved == "mps":
        logger.warning(
            "[sam3_refine] Official SAM3 has no stable MPS support; using CPU"
        )
    return "cpu"


def _to_rgb_uint8(raster_patch):
    """Convert a rasterio CHW patch to the RGB uint8 format SAM3 expects."""
    if raster_patch.ndim != 3 or raster_patch.shape[0] < 1:
        raise ValueError(f"Invalid raster patch shape: {raster_patch.shape}")
    if raster_patch.shape[0] == 1:
        chw = np.repeat(raster_patch, 3, axis=0)
    else:
        chw = raster_patch[:3]
    if chw.dtype == np.uint8:
        return np.transpose(chw, (1, 2, 0))

    rgb = np.empty(chw.shape, dtype=np.uint8)
    for band_index, band in enumerate(chw.astype(np.float32)):
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            rgb[band_index] = 0
            continue
        low, high = np.percentile(finite, (2, 98))
        if high <= low:
            rgb[band_index] = 0
        else:
            rgb[band_index] = np.clip(
                (band - low) * 255.0 / (high - low), 0, 255
            ).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0))


def load_sam3(checkpoint_path, device):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    try:
        _install_cpu_edt_fallback()
        import sam3.model_builder as sam3_model_builder
        from sam3.model.sam3_image_processor import Sam3Processor

        if device == "cpu":
            _install_cpu_builder_compatibility(sam3_model_builder)
        build_sam3_image_model = sam3_model_builder.build_sam3_image_model

        logger.info(
            f"[sam3_refine] Loading official SAM3 from {checkpoint_path} on {device}"
        )
        bpe_path = os.path.join(
            os.path.dirname(__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
        if not os.path.isfile(bpe_path):
            raise FileNotFoundError(f"SAM3 tokenizer asset not found: {bpe_path}")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu" if device == "cpu" else device,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,
        )
        model.to(device)
        model.eval()
        processor = Sam3Processor(model, device=device)
        logger.info("[sam3_refine] Official SAM3 loaded successfully")
        return model, processor
    except Exception as e:
        detail = str(e).splitlines()[0][:800]
        raise RuntimeError(
            f"Failed to load official SAM3 on {device}: "
            f"{type(e).__name__}: {detail}"
        ) from e


def _clicked_component_polygon(mask, click_px):
    from skimage.measure import find_contours

    binary = np.asarray(mask).squeeze().astype(bool)
    if binary.ndim != 2:
        raise ValueError(f"SAM3 mask must be 2D, got {binary.shape}")
    click_x, click_y = map(float, click_px)
    row = int(round(click_y))
    col = int(round(click_x))
    if row < 0 or col < 0 or row >= binary.shape[0] or col >= binary.shape[1]:
        return None
    components, _count = label(binary)
    component_id = int(components[row, col])
    if component_id <= 0:
        return None
    component = components == component_id
    rings = []
    for contour in find_contours(component.astype(np.uint8), level=0.5):
        if len(contour) < 4:
            continue
        polygon = Polygon(contour[:, [1, 0]])
        if not polygon.is_empty and polygon.is_valid and polygon.area > 0:
            rings.append(polygon)
    point = Point(click_x, click_y)
    shells = [polygon for polygon in rings if polygon.covers(point)]
    if not shells:
        return None
    shell = max(shells, key=lambda polygon: polygon.area)
    holes = [
        list(polygon.exterior.coords)
        for polygon in rings
        if polygon is not shell
        and shell.contains(polygon.representative_point())
        and not polygon.covers(point)
    ]
    result = Polygon(shell.exterior.coords, holes)
    if result.is_empty or not result.is_valid or not result.covers(point):
        return None
    return result


def predict_sam3_candidates(runtime, raster_patch, click_px, box_px=None):
    """Return only valid SAM3 masks whose connected component contains the click."""
    model, processor = runtime
    image_rgb = _to_rgb_uint8(raster_patch)
    click_x, click_y = map(float, click_px)
    state = processor.set_image(Image.fromarray(image_rgb))
    kwargs = {
        "point_coords": np.array([[click_x, click_y]], dtype=np.float32),
        "point_labels": np.array([1], dtype=np.int32),
        "multimask_output": True,
    }
    if box_px is not None:
        kwargs["box"] = np.asarray(box_px, dtype=np.float32)
    masks, scores, _ = model.predict_inst(state, **kwargs)
    if masks is None or len(masks) == 0:
        return []
    score_values = np.asarray(scores).reshape(-1) if scores is not None else np.zeros(len(masks))
    candidates = []
    for index, mask in enumerate(masks):
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        polygon = _clicked_component_polygon(mask, (click_x, click_y))
        if polygon is None:
            continue
        candidates.append({
            "geometry": polygon,
            "score": float(score_values[index]) if index < len(score_values) else 0.0,
            "mask_index": index,
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def refine_with_sam3(runtime, raster_patch, centroid_px, box_px):
    """Compatibility wrapper for the old batch caller during migration."""
    try:
        candidates = predict_sam3_candidates(
            runtime, raster_patch, centroid_px, box_px
        )
        return candidates[0]["geometry"] if candidates else None
    except Exception as e:
        logger.warning(f"[sam3_refine] SAM3 refinement failed: {e}, returning None")
        return None


def jitter_boundary(geom, max_jitter=2.0):
    coords = list(geom.exterior.coords)
    n = len(coords)
    offsets = np.random.uniform(-max_jitter, max_jitter, size=(n, 2))
    jittered = [
        (coords[i][0] + offsets[i, 0], coords[i][1] + offsets[i, 1])
        for i in range(n)
    ]
    return Polygon(jittered)


def refine_object(raster_path, gpkg_path, layer, object_id, part_id,
                  output_path, buffer_px, sam_version, sam_checkpoint="", device="cpu",
                  sam_runtime=None):
    with fiona.open(gpkg_path, layer=layer) as src:
        src_crs = src.crs
        candidate = None
        for feat in src:
            props = feat["properties"]
            if (str(props.get("object_id", "")) == object_id
                    and str(props.get("part_id", "")) == part_id):
                candidate = feat
                break

    if candidate is None:
        logger.error(
            f"[sam3_refine] Object {object_id}/{part_id} not found "
            f"in {gpkg_path} (layer={layer})"
        )
        sys.exit(1)

    props = candidate["properties"]
    class_code = int(props.get("class_code", 0))
    class_name = props.get("class_name", "未知")
    model_version = props.get("model_version", "v1.0")
    confidence_mean = float(props.get("confidence_mean", 0.0))
    confidence_std = float(props.get("confidence_std", 0.0))

    geom_shape = shape(candidate["geometry"])
    orig_bounds = geom_shape.bounds

    logger.info(
        f"[sam3_refine] Refining {object_id}/{part_id}, "
        f"class={class_code} ({class_name})"
    )

    with rasterio.open(raster_path) as raster_src:
        raster_crs = raster_src.crs
        pix_transform = raster_src.transform

        base_window = from_bounds(*orig_bounds, transform=pix_transform)
        row_min = max(0, int(np.floor(base_window.row_off)) - buffer_px)
        row_max = min(
            raster_src.height,
            int(np.ceil(base_window.row_off + base_window.height)) + buffer_px,
        )
        col_min = max(0, int(np.floor(base_window.col_off)) - buffer_px)
        col_max = min(
            raster_src.width,
            int(np.ceil(base_window.col_off + base_window.width)) + buffer_px,
        )

        if row_max <= row_min or col_max <= col_min:
            logger.warning("[sam3_refine] Object outside raster extent, using original polygon")
            refined_geom = geom_shape
        else:
            window = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)
            patch = raster_src.read(window=window)

            window_transform = rasterio.windows.transform(window, pix_transform)

            if sam_runtime is None:
                sam_runtime = load_sam3(sam_checkpoint, device)

            centroid_geo = geom_shape.centroid
            centroid_col, centroid_row = (~window_transform) * (
                centroid_geo.x, centroid_geo.y
            )
            centroid_px = (centroid_col, centroid_row)

            left_px, top_px = (~window_transform) * (
                orig_bounds[0], orig_bounds[3]
            )
            right_px, bottom_px = (~window_transform) * (
                orig_bounds[2], orig_bounds[1]
            )
            box_px = (left_px, top_px, right_px, bottom_px)

            refined_pixel = refine_with_sam3(
                sam_runtime, patch, centroid_px, box_px
            )

            if refined_pixel is not None:
                refined_pixel_coords = list(refined_pixel.exterior.coords)
                aligned_coords = []
                for px, py in refined_pixel_coords:
                    world_x, world_y = window_transform * (px, py)
                    aligned_coords.append((world_x, world_y))
                refined_geom = Polygon(aligned_coords)
                logger.info("[sam3_refine] SAM3 produced valid refinement")
            else:
                raise RuntimeError("SAM3 returned no valid mask for this object")

    sam_run_id = uuid.uuid4().hex[:8]
    refined_id = f"{object_id}_{part_id}_{sam_run_id}"

    result = {
        "geometry": mapping(refined_geom),
        "properties": {
            "refined_id": refined_id,
            "parent_object_id": object_id,
            "parent_part_id": part_id,
            "class_code": class_code,
            "class_name": class_name,
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "model_version": model_version,
            "sam_version": sam_version,
            "source": "sam3_refined",
        },
    }

    write_kwargs = dict(
        driver="GPKG", layer=LAYER_NAME, schema=SCHEMA, crs=src_crs
    )

    if os.path.exists(output_path):
        with fiona.open(output_path, "a", layer=LAYER_NAME) as dst:
            dst.write(result)
    else:
        with fiona.open(output_path, "w", **write_kwargs) as dst:
            dst.write(result)

    logger.info(
        f"[sam3_refine] Wrote refined polygon {refined_id} "
        f"→ {output_path} (layer={LAYER_NAME})"
    )
    return refined_id


def refine_all_objects(raster_path, gpkg_path, layer, output_path, buffer_px,
                       sam_version, sam_checkpoint, device):
    """Load SAM3 once and refine every semantic candidate in one process."""
    with fiona.open(gpkg_path, layer=layer) as src:
        object_keys = [
            (
                str(feat["properties"].get("object_id", "")),
                str(feat["properties"].get("part_id", "000")),
            )
            for feat in src
        ]
    if not object_keys:
        logger.info("[sam3_refine] No candidate objects to refine")
        return 0

    if os.path.exists(output_path):
        os.remove(output_path)

    runtime = load_sam3(sam_checkpoint, device)
    total = len(object_keys)
    logger.info(f"[sam3_refine] Refining {total} objects with one model load")
    for index, (object_id, part_id) in enumerate(object_keys, start=1):
        logger.info(
            f"[sam3_refine] Object {index}/{total}: {object_id}/{part_id}"
        )
        refine_object(
            raster_path, gpkg_path, layer, object_id, part_id,
            output_path, buffer_px, sam_version,
            sam_checkpoint, device, sam_runtime=runtime,
        )
    logger.info(f"[sam3_refine] Batch complete: {total}/{total} objects")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 boundary refinement for a single polygon"
    )
    parser.add_argument("--raster", required=True,
                        help="Original full raster path")
    parser.add_argument("--object_gpkg", required=True,
                        help="GeoPackage with candidate polygons")
    parser.add_argument("--object_layer", default="semantic_polygons",
                        help="Layer name in GPKG")
    parser.add_argument("--object_id", required=True,
                        help="object_id to refine")
    parser.add_argument("--part_id", default="000",
                        help="part_id to refine")
    parser.add_argument("--buffer_px", type=int, default=32,
                        help="Buffer around object in pixels")
    parser.add_argument("--output", required=True,
                        help="Output GeoPackage path")
    parser.add_argument("--sam_checkpoint", default="",
                        help="Path to SAM3 checkpoint file")
    parser.add_argument("--device", default=resolve_device("auto"),
                        help="Device for inference (auto/cpu/mps/cuda/cuda:N)")
    parser.add_argument("--sam_version", default="sam3_v1.0",
                        help="SAM3 version tag")
    args = parser.parse_args()

    try:
        args.device = resolve_sam3_device(args.device)
        if args.object_id == "__all__":
            count = refine_all_objects(
                args.raster, args.object_gpkg, args.object_layer,
                args.output, args.buffer_px, args.sam_version,
                args.sam_checkpoint, args.device,
            )
            logger.info(f"[sam3_refine] Done — refined_count={count}")
        else:
            rid = refine_object(
                args.raster, args.object_gpkg, args.object_layer,
                args.object_id, args.part_id,
                args.output, args.buffer_px, args.sam_version,
                args.sam_checkpoint, args.device,
            )
            logger.info(f"[sam3_refine] Done — refined_id={rid}")
    except Exception as e:
        logger.error(f"[sam3_refine] ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
