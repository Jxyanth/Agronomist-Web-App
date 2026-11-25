# Offline Digital Agronomist

A local-first Streamlit prototype for exploring PlantVillage leaf images, running a
locally trained MobileNetV2 classifier, and estimating fertilizer needs. Inference
does not call a cloud API.

## Setup and run (Windows PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python prepare_dataset.py
python train_model.py
streamlit run app.py
```

`prepare_dataset.py` downloads up to 200 images for each target class from the public
[PlantVillage-Dataset repository](https://github.com/spMohanty/PlantVillage-Dataset)
and caches the source images in `raw_dataset/`. It writes validated, stratified image
files into `dataset/train/`, `dataset/valid/`, and `dataset/test/` using 70/15/15
splits. To request a different maximum, run `python prepare_dataset.py
--images-per-class 300`.

If downloading fails for every class, the script creates explicitly labeled local
visual fallback images so the app can still be exercised. Those illustrations are
not real PlantVillage photos. A partial download does not silently mix fallback and
real photos; the script reports missing classes and stops.

The training step fine-tunes an ImageNet-pretrained MobileNetV2 feature classifier
and writes `models/agronomist_leaf_model.pth`, `models/class_names.json`, and
`models/training_metrics.json`. The first training run needs an internet connection
to download the pretrained weights; subsequent inference uses only the saved local
model and makes no cloud calls. The metrics include validation curves and held-out
test accuracy/per-class results. The Streamlit app can open files directly from the
local test folders or accept an upload.

## Notes

- The default four-class dataset is intentionally small. Review held-out test
  accuracy before interpreting predictions; this prototype is not a
  production-grade diagnosis model.
- Image-specific yellow/brown pixel fractions are rough color cues, not verified
  symptom segmentation. Disease guidance is associated with the predicted class,
  and low-confidence predictions suppress disease-specific treatment guidance.
- Fertilizer rates and treatment information are educational estimates. Follow
  local extension advice, pesticide labels, and soil-test recommendations.
- The upstream dataset's source and terms are available in its repository; review
  those terms before redistributing dataset images.
