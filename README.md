# Rooftop Solar Verification Pipeline – Run Instructions

This repository contains a pipeline to verify rooftop solar PV installations from satellite imagery using latitude and longitude inputs.

---

## ⚠️ GPU Requirement (Mandatory)

This pipeline **must be run on a GPU**. CPU execution is extremely slow and not recommended.

---

## 1. Requirements

- Python **3.10**
- Google Maps **Static Maps API key**
- NVIDIA GPU with **CUDA support** (e.g., T4 / RTX / A100 class)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Google Maps API Key

The pipeline fetches satellite images using Google Static Maps API.

Set the API key as an environment variable.

**Linux / macOS**

```bash
export GOOGLE_MAPS_API_KEY=your_api_key_here
```

**Windows (PowerShell)**

```powershell
setx GOOGLE_MAPS_API_KEY your_api_key_here
```

The API key is not included in this repository. Evaluators should substitute their own key.

---

## 3. Input File Location

Input must be provided as an Excel (.xlsx) file with the following columns:

| sample_id | latitude | longitude |
| --------- | -------- | --------- |

Example file location:

```
input/samples.xlsx
```

In `pipeline/main.py`, set the input file path:

```python
INPUT_XLSX = "input/samples.xlsx"
```

---

## 4. Output Location

Output locations can be configured directly in `pipeline/main.py`.

```python
ARTIFACT_DIR = "artifacts"        # replace with desired output directory for images
PREDICTION_DIR = "predictions"    # replace with desired output directory for JSON files
```

### Default Behavior

If these paths are not modified, outputs are stored in the following folders at the project root:

```
artifacts/     → satellite images and audit overlay images
predictions/   → JSON output per sample
```

Folders are created automatically if they do not already exist.

## 5. Model Files

Place the trained models in the following locations:

```
models/best.pt          # Trained YOLOv8 model
models/sam_vit_h.pth    # Segment Anything Model (SAM) weights
```

---

### SAM Weights Download

SAM weights are not included due to file size constraints.

Download manually from:

```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Place the file at:

```
models/sam_vit_h.pth
```

---

## 6. Run the Pipeline

From the project root directory, run:

```bash
python -m pipeline.main
```

The pipeline will:

- Read the Excel input file
- Fetch satellite images
- Run solar panel detection and segmentation
- Generate JSON outputs and audit overlay images

---

## 7. Output Format

For each `sample_id`, the following files are generated:

**JSON Output**

```
output/predictions/<sample_id>.json
```

**Audit Overlay Image**

```
output/artifacts/<sample_id>_overlay.jpg
```

---

## 8. Notes

- Zoom level used: 20
- Image resolution: 640×640 with scale=2 (effective 1280×1280)
- If visual evidence is insufficient (e.g., heavy shadow, tree occlusion), the site is marked as `NOT_VERIFIABLE`
