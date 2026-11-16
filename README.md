# MammoScope

A local Python desktop application for exploratory BI-RADS 1-5 classification of low-energy CDD-CESM mammograms. Upload an image, confirm its modality, and view the model prediction and five class scores. Images are processed locally.

**Research only. This model is not clinically validated and must not be used for diagnosis or treatment. Scores are not cancer probabilities. Ordinary photos, ultrasound and MRI are outside its scope.** The default model achieved 47.59% test accuracy (89/187 images), versus 33.69% for the previous head. Detailed measurements are in models/cdd_cesm_regularized/metrics.json.

## Run on this computer

Double-click START.cmd. The existing Python environment and trained model are ready locally. In Visual Studio Code, open this project folder, select the existing .venv interpreter, and run app.py or the MammoScope debug configuration.

1. The default model loads automatically; its measured test accuracy appears at the top.
2. Choose **Upload image** and select an 8-bit PNG, JPEG or BMP mammogram (maximum 30 MB and 40 million pixels). DICOM is not supported.
3. Confirm that the image is a de-identified mammogram of the training modality.
4. Select **Predict**. **Save result** exports class scores, model hash, test metrics and a timestamp as JSON.

The examples folder contains five locally available test images, one per reference class, chosen alphabetically rather than by correct prediction. The class in a filename is the expert reference label, not a guaranteed prediction.

## Fresh installation

Install Python 3.12 with the Windows Python launcher. Run `powershell -ExecutionPolicy Bypass -File SETUP.ps1`, then START.cmd. Dependencies are pinned in requirements.txt. A fresh checkout also needs a trained model bundle: model.keras and metadata.json must be in the same folder. Use **Load model folder** to load it. Large model weights and source images are excluded from Git; the current local installation includes them.

## Training

The training tab accepts a CSV with `path,birads,patient_id,split`. Paths are relative to the CSV; labels are integers 1-5 and splits are train, val or test. Every split must include all classes, and patients must not overlap. Duplicate files and equal preprocessed pixels are rejected. Training runs are written to new folders. Newly trained models load for that session; changing the default requires updating models/active_model.json.

Dataset preparation records are in data/cdd_cesm/preparation_report.json; model settings and results are in models/. Run `.venv/Scripts/python.exe -m unittest test_application -v` for software checks.

```powershell
.venv/Scripts/python.exe train.py --manifest data/manifest.csv --output models/new_cnn --epochs 10 --balanced
.venv/Scripts/python.exe train_resnet.py --manifest data/manifest.csv --output models/new_resnet --epochs 40
.venv/Scripts/python.exe improve_model.py --source models/cdd_cesm_resnet50 --output models/new_regularized
```

Model bundles use SHA-256 integrity checks, known input/output shapes and a fixed preprocessing identifier. Only load models from trusted sources; a checksum does not establish who created a model.

## Dataset source and attribution

[CDD-CESM v1 (TCIA)](https://www.cancerimagingarchive.net/collection/cdd-cesm/) is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite [Khaled et al., dataset (2021)](https://doi.org/10.7937/29kw-ae92), [Khaled et al. (2022)](https://doi.org/10.1038/s41597-022-01238-0) and [Clark et al. (2013)](https://doi.org/10.1007/s10278-013-9622-7).

Images came from the [Kaggle mirror](https://www.kaggle.com/datasets/hanaraafatt/cdd-cesm/versions/1); annotations came from [TCIA](https://www.cancerimagingarchive.net/wp-content/uploads/Radiology-manual-annotations.xlsx). The subset uses 931 low-energy DM images from 316 patients, excluding multiple-valued labels and BI-RADS 6. Filenames have spaces removed; images are converted to grayscale, resized to 224 by 224 and divided by 255. Providers do not endorse this application. Training, validation and test partitions are patient-disjoint. Evaluation is exploratory, not external clinical validation.

```powershell
.venv/Scripts/python.exe prepare_cdd.py --archive PATH_TO_ARCHIVE.zip --annotations PATH_TO_ANNOTATIONS.xlsx
```
