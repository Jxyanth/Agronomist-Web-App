# Offline Digital Agronomist

A local-first Streamlit prototype for exploring PlantVillage leaf images, running a
locally trained MobileNetV2 classifier, and estimating fertilizer needs. Inference
does not call a cloud API.

## Setup and run (Windows PowerShell)

Use 64-bit Python 3.12 for this project. Open PowerShell in this project folder.
The following commands replace any existing `.venv` (which may contain packages
installed for a different Python version), then create a clean environment and
install the dependencies:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The commands above use the virtual environment's Python directly, so you do not
need to activate it. This also avoids PowerShell execution-policy errors from
running `Activate.ps1` and ensures Streamlit is run from the environment where
the dependencies were installed. If the Python 3.12 launcher command is not
available, install 64-bit Python 3.12 and rerun the setup commands.

The repository includes a dataset and a trained model. To recreate the dataset
or train a new model, run these optional commands before starting the app:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
.\.venv\Scripts\python.exe train_model.py
```

`prepare_dataset.py` downloads up to 80 images for each target class from the public
[PlantVillage-Dataset repository](https://github.com/spMohanty/PlantVillage-Dataset)
and caches the source images in `raw_dataset/`. It writes validated, stratified image
files into `dataset/train/`, `dataset/valid/`, and `dataset/test/` using 70/15/15
splits. To request a different maximum, run `python prepare_dataset.py
--images-per-class 120`. If the source throttles downloads before that count is
reached, the script stops without replacing the existing dataset split.

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
