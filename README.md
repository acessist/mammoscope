# MammoScope

A local Python desktop application for exploratory BI-RADS 1–5 classification of low-energy CDD-CESM mammograms. Upload an image, confirm its modality, and view the model prediction and five class scores. Images are processed locally.

**Research only. This model is not clinically validated and must not be used for diagnosis or treatment. Scores are not cancer probabilities. Ordinary photos, ultrasound and MRI are outside its scope.** See RESULTS.md for measured performance.

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

The training tab accepts a CSV with `path,birads,patient_id,split`. Paths are relative to the CSV; labels are integers 1–5 and splits are train, val or test. Every split must include all classes, and patients must not overlap. Duplicate files and equal preprocessed pixels are rejected. Training runs are written to new folders. Newly trained models load for that session; changing the default requires updating models/active_model.json.

See DATA.md for dataset preparation, METHODS.md for the manuscript appendix, RESULTS.md for experiments, and TEST_REPORT.md for checks. CODE_APPENDIX.txt contains the application source snapshot.

```powershell
.venv/Scripts/python.exe train.py --manifest data/manifest.csv --output models/new_cnn --epochs 10 --balanced
.venv/Scripts/python.exe train_resnet.py --manifest data/manifest.csv --output models/new_resnet --epochs 40
.venv/Scripts/python.exe improve_model.py --source models/cdd_cesm_resnet50 --output models/new_regularized
```

Model bundles use SHA-256 integrity checks, known input/output shapes and a fixed preprocessing identifier. Only load models from trusted sources; a checksum does not establish who created a model.

