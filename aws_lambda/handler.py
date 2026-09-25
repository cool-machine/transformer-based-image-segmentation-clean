"""AWS Lambda backend for the OCP8 image-segmentation recruiting demo."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

BUCKET = os.environ.get("OCP8_BUCKET", "clarifiance-ocp8-artifacts-174208891400")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/segformer/")
IMAGES_PREFIX = os.environ.get("IMAGES_PREFIX", "images1/images/")
MASKS_PREFIX = os.environ.get("MASKS_PREFIX", "images1/masks/")
MODEL_DIR = Path("/tmp/ocp8-model/segformer")
DISPLAY_SIZE = (1024, 512)

S3 = boto3.client("s3")
_MODEL = None

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

COLOR_MAP = {
    0: (31, 119, 180),
    1: (214, 39, 40),
    2: (255, 127, 14),
    3: (44, 160, 44),
    4: (148, 103, 189),
    5: (140, 86, 75),
    6: (227, 119, 194),
    7: (23, 190, 207),
}

CITYSCAPES_TO_8CLASS = {
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 1, 5: 1,
    6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2,
    14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3,
    20: 4, 21: 4, 22: 4, 23: 4,
    24: 5, 25: 5,
    26: 6,
    27: 7, 28: 7, 29: 7,
}


def _response(status_code: int, payload: dict[str, Any] | str) -> dict[str, Any]:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", **CORS_HEADERS},
        "body": body,
        "isBase64Encoded": False,
    }


def _query(event: dict[str, Any], name: str) -> str | None:
    values = event.get("queryStringParameters") or {}
    return values.get(name)


def _path(event: dict[str, Any]) -> str:
    path = event.get("rawPath") or event.get("path") or "/"
    if path.startswith("/api/"):
        path = path[4:]
    return path.rstrip("/") or "/"


def _s3_bytes(key: str) -> bytes:
    return S3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def _image_key(image_name: str) -> str:
    clean = Path(image_name).name.removesuffix(".png").removesuffix(".jpg")
    if not clean.endswith("_leftImg8bit"):
        clean += "_leftImg8bit"
    return f"{IMAGES_PREFIX}{clean}.png"


def _mask_key(image_name: str) -> str:
    clean = Path(image_name).name.removesuffix(".png").removesuffix(".jpg")
    clean = clean.removesuffix("_leftImg8bit")
    return f"{MASKS_PREFIX}{clean}_gtFine_labelIds.png"


def _download_model() -> Path:
    config_path = MODEL_DIR / "config.json"
    weights_path = MODEL_DIR / "tf_model.h5"
    if config_path.exists() and weights_path.exists():
        return MODEL_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tf_model.h5"):
        S3.download_file(BUCKET, f"{MODEL_PREFIX}{filename}", str(MODEL_DIR / filename))
    return MODEL_DIR


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from transformers import TFSegformerForSemanticSegmentation

    model_dir = _download_model()
    LOGGER.info("Loading the fine-tuned SegFormer model from S3 artifacts")
    _MODEL = TFSegformerForSemanticSegmentation.from_pretrained(str(model_dir))
    return _MODEL


def _predict(image_array: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    model = _load_model()
    original_height, original_width = image_array.shape[:2]
    resized = tf.image.resize(image_array, [512, 1024])
    normalized = tf.cast(resized, tf.float32) / 255.0
    batch = tf.expand_dims(normalized, 0)

    try:
        outputs = model(batch, training=False)
    except Exception:
        outputs = model(tf.transpose(batch, [0, 3, 1, 2]), training=False)

    logits = outputs.logits
    if len(logits.shape) == 4 and logits.shape[1] == 8:
        logits = tf.transpose(logits, [0, 2, 3, 1])
    mask = tf.squeeze(tf.argmax(logits, axis=-1))
    mask = tf.image.resize(
        tf.expand_dims(tf.cast(mask, tf.float32), -1),
        [original_height, original_width],
        method="nearest",
    )
    return tf.squeeze(mask).numpy().astype(np.uint8)


def _to_eight_classes(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = mask[:, :, 0] if mask.shape[-1] == 1 else mask.squeeze()
    converted = np.zeros_like(mask, dtype=np.uint8)
    for source, target in CITYSCAPES_TO_8CLASS.items():
        converted[mask == source] = target
    return converted


def _colorize(mask: np.ndarray) -> Image.Image:
    mask8 = _to_eight_classes(mask)
    colored = np.zeros((*mask8.shape, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        colored[mask8 == class_id] = color
    return Image.fromarray(colored, mode="RGB")


def _display_image(image: Image.Image) -> Image.Image:
    rendered = image.convert("RGB")
    rendered.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS)
    return rendered


def _overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = _display_image(image).convert("RGBA")
    layer = mask.resize(base.size, Image.Resampling.NEAREST).convert("RGBA")
    return Image.blend(base, layer, 0.8).convert("RGB")


def _data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _health() -> dict[str, Any]:
    return _response(200, {
        "status": "ok",
        "message": "AWS Lambda OCP8 visualization stack",
        "version": "aws-1.0",
        "python_version": sys.version.split()[0],
        "storage_configured": bool(BUCKET),
        "model_loaded": _MODEL is not None,
        "features": ["Amazon S3 access", "PIL image processing", "SegFormer inference"],
    })


def _images() -> dict[str, Any]:
    paginator = S3.get_paginator("list_objects_v2")
    found = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=IMAGES_PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.lower().endswith((".png", ".jpg", ".jpeg")):
                found.append({
                    "name": key.removeprefix("images1/"),
                    "size": item["Size"],
                    "last_modified": item["LastModified"].isoformat(),
                })
    found.sort(key=lambda item: item["name"])
    return _response(200, {
        "images": found[:10],
        "total_count": len(found),
        "container": BUCKET,
        "path": "images/",
    })


def _thumbnail(image_name: str | None) -> dict[str, Any]:
    if not image_name:
        return _response(400, {"error": "image_name parameter required"})
    image = Image.open(BytesIO(_s3_bytes(_image_key(image_name))))
    image.thumbnail((300, 200), Image.Resampling.LANCZOS)
    return _response(200, {
        "status": "success",
        "image_name": image_name,
        "thumbnail": _data_url(image.convert("RGB")),
    })


def _colorized_masks(image_name: str | None) -> dict[str, Any]:
    image_name = image_name or "lindau_000000_000019_leftImg8bit"
    image = Image.open(BytesIO(_s3_bytes(_image_key(image_name)))).convert("RGB")
    mask = Image.open(BytesIO(_s3_bytes(_mask_key(image_name))))
    image_array = np.asarray(image)
    mask_array = np.asarray(mask)

    prediction = _predict(image_array)
    result = {
        "status": "success",
        "message": "Individual visualizations generated successfully",
        "image_name": image_name,
        "image_shape": list(image_array.shape),
        "mask_shape": list(mask_array.shape),
        "mask_unique_values": sorted(np.unique(mask_array).tolist()),
        "visualizations": {
            "original": _data_url(_display_image(image)),
            "ground_truth": _data_url(_overlay(image, _colorize(mask_array))),
            "predicted": _data_url(_overlay(image, _colorize(prediction))),
        },
        "generation_method": "AWS Lambda SegFormer inference",
    }
    return _response(200, result)


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return _response(200, "")

    try:
        path = _path(event)
        if path == "/health":
            return _health()
        if path == "/images":
            return _images()
        if path == "/image-thumbnail":
            return _thumbnail(_query(event, "image_name"))
        if path == "/colorized-masks":
            return _colorized_masks(_query(event, "image_name"))
        return _response(404, {"error": "Not found", "path": path})
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            return _response(404, {"error": "Requested image or mask was not found"})
        LOGGER.exception("OCP8 S3 request failed")
        return _response(500, {"error": "Storage request failed"})
    except Exception as exc:
        LOGGER.exception("OCP8 request failed")
        return _response(500, {"error": str(exc)})
