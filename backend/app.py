"""
Prototype local FastAPI pour la segmentation semantique de scenes urbaines.
Ce serveur a ete utilise pour le developpement et la validation locale
avant la migration vers Azure Functions (function_app.py) en production.

Lancement : uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import logging
import time
import os
import io

# --- Configuration -----------------------------------------------------------

app = FastAPI(
    title="Segmentation Semantique - API Locale",
    description="API de segmentation d'images urbaines (Cityscapes 8 classes). "
                "Prototype local FastAPI - equivalent fonctionnel de l'API Azure Functions.",
    version="1.0.0",
)

# CORS - autoriser le frontend GitHub Pages et le developpement local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cool-machine.github.io",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constantes --------------------------------------------------------------

# Palette de couleurs identique a la production (function_app.py)
BEAUTIFUL_COLOR_MAP = {
    0: (255, 127, 14),     # flat/road - ORANGE (was vehicle)
    1: (214, 39, 40),      # human/person - ROUGE
    2: (148, 103, 189),    # vehicle - VIOLET
    3: (44, 160, 44),      # construction/building - VERT
    4: (31, 119, 180),     # object/pole/sign - BLEU
    5: (140, 86, 75),      # nature/vegetation - BRUN
    6: (227, 119, 194),    # sky - ROSE
    7: (23, 190, 207),     # void/other - CYAN
}

COLOR_MAP_ARRAY = np.array([BEAUTIFUL_COLOR_MAP[i] for i in range(8)], dtype=np.uint8)

CLASS_NAMES = {
    0: "flat", 1: "human", 2: "vehicle", 3: "construction",
    4: "object", 5: "nature", 6: "sky", 7: "void",
}

# Mapping Cityscapes 30 classes -> 8 classes
CITYSCAPES_TO_8CLASS = {
    0: 0, 1: 0, 2: 0, 3: 0,           # road, sidewalk, parking, rail track -> flat
    4: 1, 5: 1,                         # person, rider -> human
    6: 2, 7: 2, 8: 2, 9: 2,           # car, truck, bus, on rails -> vehicle
    10: 2, 11: 2, 12: 2, 13: 2,       # motorcycle, bicycle, caravan, trailer -> vehicle
    14: 3, 15: 3, 16: 3, 17: 3,       # building, wall, fence, guard rail -> construction
    18: 3, 19: 3,                       # bridge, tunnel -> construction
    20: 4, 21: 4, 22: 4, 23: 4,       # pole, pole group, traffic sign, traffic light -> object
    24: 5, 25: 5,                       # vegetation, terrain -> nature
    26: 6,                              # sky -> sky
    27: 7, 28: 7, 29: 7,              # ground, dynamic, static -> void
}

# Normalisation ImageNet (identique au pipeline d'entrainement)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Repertoire des images locales (configurable)
IMAGE_DIR = os.environ.get("IMAGE_DIR", "images")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# --- Chargement des modeles (lazy loading) ------------------------------------

_models = {}


def _get_model(name: str):
    """Charge un modele en memoire au premier appel, puis le met en cache."""
    if name not in _models:
        logger.info(f"Chargement du modele {name}...")
        t0 = time.time()
        if name == "segformer":
            try:
                from transformers import TFSegformerForSemanticSegmentation, SegformerConfig
                config = SegformerConfig(
                    num_labels=8,
                    id2label=CLASS_NAMES,
                    label2id={v: k for k, v in CLASS_NAMES.items()},
                    image_size=(512, 1024),
                )
                model = TFSegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
                    config=config,
                    ignore_mismatched_sizes=True,
                )
                # Charger les poids fine-tunes si disponibles
                weights_path = os.path.join(MODEL_DIR, "segformer_b0_cityscapes.h5")
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    logger.info(f"Poids fine-tunes charges depuis {weights_path}")
                _models[name] = model
            except ImportError:
                raise HTTPException(500, "transformers non installe")
        elif name == "unet":
            weights_path = os.path.join(MODEL_DIR, "unet_vgg16_cityscapes.h5")
            if os.path.exists(weights_path):
                _models[name] = tf.keras.models.load_model(weights_path)
            else:
                raise HTTPException(500, f"Poids UNet introuvables: {weights_path}")
        else:
            raise HTTPException(400, f"Modele inconnu: {name}")
        logger.info(f"Modele {name} charge en {time.time() - t0:.1f}s")
    return _models[name]


# --- Fonctions utilitaires ----------------------------------------------------

def _load_image(image_name: str) -> np.ndarray:
    """Charge et preprocesse une image depuis le repertoire local."""
    for ext in [".png", ".jpg", ".jpeg", ""]:
        path = os.path.join(IMAGE_DIR, image_name + ext)
        if os.path.exists(path):
            break
    else:
        raise HTTPException(404, f"Image introuvable: {image_name}")

    img = Image.open(path).convert("RGB")
    img = img.resize((1024, 512), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def _predict(image: np.ndarray, model_name: str) -> np.ndarray:
    """Execute l'inference et retourne le masque 8 classes."""
    model = _get_model(model_name)
    batch = np.expand_dims(image, axis=0)

    if model_name == "segformer":
        outputs = model(tf.constant(batch), training=False)
        logits = outputs.logits[0].numpy()
        # Upsample logits to input resolution
        logits_upsampled = tf.image.resize(
            logits[np.newaxis], (512, 1024), method="bilinear"
        )[0].numpy()
        mask = np.argmax(logits_upsampled, axis=-1).astype(np.uint8)
    else:
        logits = model.predict(batch, verbose=0)[0]
        mask = np.argmax(logits, axis=-1).astype(np.uint8)

    return mask


def _colorize(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque 8 classes en image RGB colorisee."""
    return COLOR_MAP_ARRAY[mask]


# --- Endpoints ----------------------------------------------------------------

@app.get("/health")
def health():
    """Verification de sante - retourne le statut des modeles disponibles."""
    return {
        "status": "ok",
        "models": ["segformer", "unet"],
        "models_loaded": list(_models.keys()),
        "image_dir": IMAGE_DIR,
    }


@app.get("/images")
def list_images():
    """Liste les images Cityscapes disponibles localement."""
    if not os.path.isdir(IMAGE_DIR):
        return {"images": [], "error": f"Repertoire {IMAGE_DIR} introuvable"}
    images = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    return {"images": images, "count": len(images)}


@app.get("/image-thumbnail")
def image_thumbnail(image_name: str = Query(..., description="Nom de l'image")):
    """Retourne une miniature JPEG de l'image demandee."""
    for ext in [".png", ".jpg", ".jpeg", ""]:
        path = os.path.join(IMAGE_DIR, image_name + ext)
        if os.path.exists(path):
            break
    else:
        raise HTTPException(404, f"Image introuvable: {image_name}")

    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 128), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.post("/segment")
def segment(
    image_name: str = Query(..., description="Nom de l'image"),
    model: str = Query("segformer", description="Modele: segformer ou unet"),
):
    """Segmente une image et retourne le masque en base64 + metriques."""
    t0 = time.time()
    image = _load_image(image_name)
    mask = _predict(image, model)
    inference_time = time.time() - t0

    # Coloriser et encoder
    colorized = _colorize(mask)
    img = Image.fromarray(colorized)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Classes detectees
    unique_classes = sorted(int(c) for c in np.unique(mask))

    return JSONResponse({
        "mask_b64": mask_b64,
        "model": model,
        "image_name": image_name,
        "inference_time": round(inference_time, 3),
        "classes_detected": unique_classes,
        "class_names": {str(c): CLASS_NAMES[c] for c in unique_classes},
    })


@app.get("/colorized-masks")
def colorized_masks(image_name: str = Query(..., description="Nom de l'image")):
    """Retourne le masque colorise en PNG (endpoint principal du frontend)."""
    image = _load_image(image_name)
    mask = _predict(image, "segformer")
    colorized = _colorize(mask)

    img = Image.fromarray(colorized)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# --- Point d'entree -----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
