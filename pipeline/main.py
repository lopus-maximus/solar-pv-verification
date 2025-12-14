import os
import cv2
import json
import pandas as pd
import torch

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from pipeline.fetch_image import fetch_static_map
from pipeline.classify import classify_solar
from pipeline.sam_area import compute_area
from pipeline.qc import qc_check
from pipeline.overlay import draw_overlay


# ---------- CONFIG ----------
INPUT_XLSX = "PASTE_INPUT_FILE_LOCATION_HERE"
ARTIFACT_DIR = "artifacts"  #replace with location of output directory
PREDICTION_DIR = "predictions"   #replace with location of output directory for JSON data

MODEL_PATH = "models/best.pt"
SAM_PATH = "models/sam_vit_h.pth"


# ---------- Create output folders ----------
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


# ---------- Load models ----------
yolo = YOLO(MODEL_PATH)

sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)


def process_site(sample_id, lat, lon):
    img_path = f"{ARTIFACT_DIR}/{sample_id}.jpg"
    overlay_path = f"{ARTIFACT_DIR}/{sample_id}_overlay.jpg"

    # 1. Fetch image
    fetch_static_map(lat, lon, img_path)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Image read failed for sample {sample_id}")
        return

    # 2. YOLO classification
    has_solar, conf, buffer_used, boxes = classify_solar(
        yolo, img, lat
    )

    mask = None
    area = 0.0

    # 3. SAM segmentation + area
    if has_solar and boxes:
        predictor.set_image(img)
        masks, _, _ = predictor.predict(
            box=boxes[0],
            multimask_output=False
        )
        mask = masks[0]
        area = compute_area(mask, lat)

    # 4. Create audit overlay
    draw_overlay(
        image=img,
        boxes=boxes,
        mask=mask,
        out_path=overlay_path
    )

    # 5. Save JSON output
    result = {
        "sample_id": int(sample_id),
        "lat": float(lat),
        "lon": float(lon),
        "has_solar": bool(has_solar),
        "confidence": float(conf),
        "pv_area_sqm_est": float(area),
        "buffer_radius_sqft": buffer_used,
        "qc_status": qc_check(img),
        "bbox_or_mask": "SAM_MASK" if mask is not None else "YOLO_BBOX",
        "image_metadata": {
            "source": "Google Static Maps",
            "zoom": 20,
        }
    }

    with open(f"{PREDICTION_DIR}/{sample_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------- MAIN ----------
if __name__ == "__main__":

    df = pd.read_excel(INPUT_XLSX)

    for _, row in df.iterrows():
        print(f"Processing sample {row['sample_id']}")
        process_site(
            sample_id=row["sample_id"],
            lat=row["latitude"],
            lon=row["longitude"]
        )
